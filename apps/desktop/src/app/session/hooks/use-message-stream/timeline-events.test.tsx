import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { chatMessageText, toChatMessages } from '@/lib/chat-messages'

import { reconcileDurableHistory } from '../use-session-actions/utils'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'timeline-session'

let stream: MessageStreamHarness

const event = (type: GatewayEventName, timestamp: number, payload: Record<string, unknown> = {}) =>
  act(() => stream.handleEvent({ payload: { ...payload, timestamp }, session_id: SID, type }))

describe('live transcript timeline events', () => {
  beforeEach(async () => {
    stream = renderMessageStream(SID)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('records commentary, tool, resumed text, and turn-stop boundaries', () => {
    event('message.start', 100)
    event('message.delta', 101.125, { text: 'Let me inspect it.' })
    event('message.interim', 101.75, { already_streamed: true, text: 'Let me inspect it.' })
    event('tool.start', 102.25, { args: { path: 'README.md' }, name: 'read_file', tool_id: 'call-1' })
    event('tool.complete', 104.5, { name: 'read_file', result: { content: 'ok' }, tool_id: 'call-1' })
    event('message.delta', 105.625, { text: 'The file looks good.' })
    event('message.complete', 106.875, { text: 'The file looks good.' })

    const assistants = stream.state(SID).messages.filter(message => message.role === 'assistant') ?? []

    expect(assistants).toHaveLength(2)
    expect([assistants[0].timestamp, assistants[0].completedAt]).toEqual([101.125, 101.75])
    expect(assistants[0].parts.map(part => [part.timestamp, part.completedAt])).toEqual([[101.125, 101.75]])

    expect([assistants[1].timestamp, assistants[1].completedAt]).toEqual([102.25, 106.875])
    expect(assistants[1].parts.map(part => part.type)).toEqual(['tool-call', 'text'])
    expect(assistants[1].parts.map(part => [part.timestamp, part.completedAt])).toEqual([
      [102.25, 104.5],
      [105.625, 106.875]
    ])
  })

  it('preserves cross-channel delta order inside one flush window', () => {
    event('message.start', 200)
    event('reasoning.delta', 201.125, { text: 'Thinking first.' })
    event('message.delta', 202.25, { text: 'Then speaking.' })
    event('tool.start', 203.5, { args: {}, name: 'terminal', tool_id: 'call-2' })

    const assistant = stream.state(SID).messages.find(message => message.role === 'assistant')

    expect(assistant?.parts.map(part => part.type)).toEqual(['reasoning', 'text', 'tool-call'])
    expect(assistant?.parts.map(part => part.timestamp)).toEqual([201.125, 202.25, 203.5])
  })

  it('uses the gateway event time for an error boundary', () => {
    event('message.start', 300)
    event('error', 301.875, { error: 'provider failed' })

    const assistant = stream.state(SID).messages.find(message => message.role === 'assistant')

    expect(assistant?.error).toBeTruthy()
    expect([assistant?.timestamp, assistant?.completedAt]).toEqual([301.875, 301.875])
  })

  it.each([SID, 'background-session'])('keeps a full model-switch warning in its owning session %s', sessionId => {
    event('message.start', 450)
    if (sessionId !== SID) {
      act(() => stream.handleEvent({ session_id: sessionId, type: 'message.start', payload: { timestamp: 450 } }))
    }
    const focusedBefore = stream.state(SID)
    const before = stream.state(sessionId)
    const text =
      '  PROVIDER AUTOMATICALLY CHANGED: openai-codex -> openrouter. Additional costs may apply.\n\n' +
      'Catalog validation notice. Keep <b>literal markup</b> and \u001b[31mcontrol text\u001b[0m.\n'
    expect(before.busy).toBe(true)

    act(() =>
      stream.handleEvent({
        session_id: sessionId,
        type: 'status.update',
        payload: { kind: 'model-switch-warning', text, timestamp: 451.625 }
      })
    )

    const after = stream.state(sessionId)
    const system = after.messages.at(-1)
    expect(after.messages).toHaveLength(before.messages.length + 1)
    expect(system).toMatchObject({
      role: 'system',
      parts: [{ type: 'text', text: `warning: ${text}`, timestamp: 451.625 }],
      timestamp: 451.625
    })
    expect({ ...after, messages: before.messages }).toEqual(before)
    expect(after.messages.slice(0, -1)).toEqual(before.messages)
    if (sessionId !== SID) {
      expect(stream.state(SID)).toEqual(focusedBefore)
    }

    act(() =>
      stream.handleEvent({
        session_id: sessionId,
        type: 'status.update',
        payload: { kind: 'progress', text: 'Working...', timestamp: 452 }
      })
    )
    expect(stream.state(sessionId)).toEqual(after)
  })

  it('keeps one full local warning through repeated stored-history refreshes without changing turn state', () => {
    event('message.start', 450)
    const text =
      '  PROVIDER AUTOMATICALLY CHANGED: openai-codex -> openrouter. Additional costs may apply.\n\n' +
      'Catalog validation notice. Keep <b>literal markup</b> and \u001b[31mcontrol text\u001b[0m.\n'
    event('status.update', 451.625, { kind: 'model-switch-warning', text })

    const afterEvent = stream.state(SID)
    const otherSession = stream.state('another-session')
    stream.states.set('another-session', otherSession)
    expect(afterEvent.busy).toBe(true)
    stream.states.set(SID, {
      ...afterEvent,
      messages: [
        ...afterEvent.messages,
        { id: 'stale-system', role: 'system', parts: [{ type: 'text', text: 'old status' }] }
      ]
    })
    const stored = toChatMessages([
      { id: 70, role: 'user', content: 'Give the answer.' },
      { id: 71, role: 'assistant', content: 'The stored answer.' }
    ])

    for (const _ of [1, 2]) {
      const current = stream.state(SID)
      stream.states.set(SID, { ...current, messages: reconcileDurableHistory(stored, current.messages) })
      const refreshed = stream.state(SID)
      const warnings = refreshed.messages.filter(message => message.role === 'system')
      expect(warnings).toHaveLength(1)
      expect(chatMessageText(warnings[0])).toBe(`warning: ${text}`)
      expect(warnings[0].id).toBe(afterEvent.messages.at(-1)?.id)
      expect({ ...refreshed, messages: afterEvent.messages }).toEqual(afterEvent)
      expect(stream.state('another-session')).toEqual(otherSession)
    }

    const represented = [
      ...stored,
      { ...stream.state(SID).messages.find(message => message.role === 'system')!, id: 'stored-warning' }
    ]
    const reconciled = reconcileDurableHistory(represented, stream.state(SID).messages)
    expect(reconciled.filter(message => message.role === 'system')).toHaveLength(1)
    expect(reconciled.at(-1)?.id).toBe('stored-warning')

    const superseded = [
      ...stored,
      {
        ...represented.at(-1)!,
        id: afterEvent.messages.at(-1)!.id,
        parts: [{ type: 'text' as const, text: 'warning: a newer notice replaces this one' }]
      }
    ]
    expect(
      reconcileDurableHistory(superseded, stream.state(SID).messages).filter(message => message.role === 'system')
    ).toEqual([superseded.at(-1)])
  })

  it('retains distinct identical warnings across A-to-B, B-to-A, A-to-B turns', () => {
    const warning = 'Provider automatically changed: A -> B. Additional costs may apply.'
    const stored = toChatMessages([
      { id: 80, role: 'user', content: 'Use B.' },
      { id: 81, role: 'assistant', content: 'Using B.' },
      { id: 82, role: 'user', content: 'Use A.' },
      { id: 83, role: 'assistant', content: 'Using A.' },
      { id: 84, role: 'user', content: 'Use B again.' },
      { id: 85, role: 'assistant', content: 'Using B again.' }
    ])
    for (const [index, text] of [warning, 'Provider automatically changed: B -> A.', warning].entries()) {
      const turn = index * 10 + 600
      stream.states.set(SID, {
        ...stream.state(SID),
        messages: [...stream.state(SID).messages, stored[index * 2]]
      })
      event('message.start', turn)
      event('status.update', turn + 1, { kind: 'model-switch-warning', text })
      event('message.delta', turn + 2, { text: chatMessageText(stored[index * 2 + 1]) })
      event('message.complete', turn + 3, { text: chatMessageText(stored[index * 2 + 1]) })
    }

    const localWarnings = stream.state(SID).messages.filter(message => message.role === 'system')
    expect(localWarnings.map(message => message.timestamp)).toEqual([601, 611, 621])
    expect(new Set(localWarnings.map(message => message.id)).size).toBe(3)

    for (const representedIndex of [0, 2]) {
      const represented = { ...localWarnings[representedIndex], id: 'stored-warning' }
      const history = stored.flatMap((row, index) => (index === representedIndex * 2 ? [row, represented] : [row]))
      let refreshed = reconcileDurableHistory(history, stream.state(SID).messages)
      for (const _ of [1, 2]) {
        expect(refreshed.map(message => message.role)).toEqual([
          'user',
          'system',
          'assistant',
          'user',
          'system',
          'assistant',
          'user',
          'system',
          'assistant'
        ])
        expect(refreshed.filter(message => message.role === 'system').map(message => message.timestamp)).toEqual([
          601, 611, 621
        ])
        expect(refreshed.filter(message => message.role === 'system').map(message => message.id)).toEqual(
          localWarnings.map((warning, index) => (index === representedIndex ? 'stored-warning' : warning.id))
        )
        refreshed = reconcileDurableHistory(history, refreshed)
      }
    }
  })

  it('does not reuse a warning id when a same-millisecond refresh removes the prior transient row', () => {
    vi.spyOn(Date, 'now').mockReturnValue(1_000)
    event('message.start', 800)
    event('status.update', 801, { kind: 'model-switch-warning', text: 'First warning.' })
    const first = stream.state(SID).messages.find(message => message.role === 'system')!

    const current = stream.state(SID)
    stream.states.set(SID, { ...current, messages: current.messages.filter(message => message !== first) })
    event('status.update', 802, { kind: 'model-switch-warning', text: 'Second warning.' })
    const second = stream.state(SID).messages.find(message => message.role === 'system')!

    expect(second.id).not.toBe(first.id)
  })

  it('matches identical row-id-less prompts by turn occurrence instead of text or reused client id', () => {
    const durable = toChatMessages([
      { id: 90, role: 'user', content: 'Repeat this prompt.' },
      { id: 91, role: 'assistant', content: 'First answer.' },
      { id: 92, role: 'user', content: 'Repeat this prompt.' },
      { id: 93, role: 'assistant', content: 'Second answer.' }
    ])
    const firstUser = { ...durable[0], id: 'reused-user', rowId: undefined }
    const secondUser = { ...durable[2], id: 'reused-user', rowId: undefined }
    const warningText = 'warning: Provider automatically changed: A -> B.'
    const firstWarning = {
      id: 'model-switch-warning-first',
      role: 'system' as const,
      parts: [{ type: 'text' as const, text: warningText }]
    }
    const secondWarning = { ...firstWarning, id: 'model-switch-warning-second' }
    const previous = [firstUser, firstWarning, durable[1], secondUser, secondWarning, durable[3]]
    const storedSecond = { ...secondWarning, id: 'stored-second-warning' }
    const next = [firstUser, durable[1], secondUser, storedSecond, durable[3]]

    const reconciled = reconcileDurableHistory(next, previous)
    expect(reconciled.map(message => message.role)).toEqual([
      'user',
      'system',
      'assistant',
      'user',
      'system',
      'assistant'
    ])
    expect(reconciled.filter(message => message.role === 'system').map(message => message.id)).toEqual([
      firstWarning.id,
      storedSecond.id
    ])
  })

  it('keeps a warning with its shifted turn when duplicate prompt identities surround it', () => {
    const prior = toChatMessages([
      { id: 100, role: 'user', content: 'Same prompt.' },
      { id: 101, role: 'assistant', content: 'Answer A.' },
      { id: 102, role: 'user', content: 'Same prompt.' },
      { id: 103, role: 'assistant', content: 'Answer B.' }
    ])
    const next = toChatMessages([
      { id: 102, role: 'user', content: 'Same prompt.' },
      { id: 103, role: 'assistant', content: 'Answer B.' },
      { id: 104, role: 'user', content: 'Same prompt.' },
      { id: 105, role: 'assistant', content: 'Answer C.' }
    ])
    const reuseIdentity = (message: (typeof prior)[number]) => ({ ...message, id: 'reused-user', rowId: undefined })
    const localWarning = {
      id: 'model-switch-warning-shifted',
      role: 'system' as const,
      parts: [{ type: 'text' as const, text: 'warning: shifted turn' }]
    }
    const storedWarning = { ...localWarning, id: 'stored-shifted-warning' }
    const previous = [reuseIdentity(prior[0]), prior[1], reuseIdentity(prior[2]), localWarning, prior[3]]
    const refreshed = [reuseIdentity(next[0]), storedWarning, next[1], reuseIdentity(next[2]), next[3]]

    const reconciled = reconcileDurableHistory(refreshed, previous)
    expect(reconciled.filter(message => message.role === 'system')).toEqual([storedWarning])
    expect(reconciled.indexOf(storedWarning)).toBe(1)
  })

  it('does not let duplicate local warning ids claim the same stored occurrence', () => {
    const stored = toChatMessages([
      { id: 110, role: 'user', content: 'First prompt.' },
      { id: 111, role: 'assistant', content: 'First answer.' },
      { id: 112, role: 'user', content: 'Second prompt.' },
      { id: 113, role: 'assistant', content: 'Second answer.' }
    ])
    const duplicateId = 'model-switch-warning-duplicate'
    const firstWarning = {
      id: duplicateId,
      role: 'system' as const,
      parts: [{ type: 'text' as const, text: 'warning: repeated warning' }]
    }
    const secondWarning = { ...firstWarning }
    const representedFirst = { ...firstWarning, id: duplicateId }
    const previous = [stored[0], firstWarning, stored[1], stored[2], secondWarning, stored[3]]
    const next = [stored[0], representedFirst, stored[1], stored[2], stored[3]]

    const reconciled = reconcileDurableHistory(next, previous)
    expect(reconciled.map(message => message.role)).toEqual([
      'user',
      'system',
      'assistant',
      'user',
      'system',
      'assistant'
    ])
    expect(reconciled.filter(message => message.role === 'system')).toEqual([representedFirst, secondWarning])
  })

  it('keeps the local warning before its stable answer when its owning user row was truncated', () => {
    const stored = toChatMessages([
      { id: 120, role: 'user', content: 'First prompt.' },
      { id: 121, role: 'assistant', content: 'First answer.' },
      { id: 122, role: 'user', content: 'Next prompt.' },
      { id: 123, role: 'assistant', content: 'Next answer.' }
    ])
    const localWarning = {
      id: 'model-switch-warning-truncated',
      role: 'system' as const,
      parts: [{ type: 'text' as const, text: 'warning: Provider automatically changed.' }]
    }
    const previous = [
      stored[0],
      localWarning,
      stored[1],
      { id: 'unrelated-status', role: 'system' as const, parts: [{ type: 'text' as const, text: 'old status' }] },
      stored[2],
      stored[3]
    ]
    const truncated = stored.slice(1)
    let refreshed = reconcileDurableHistory(truncated, previous)
    for (const _ of [1, 2]) {
      expect(refreshed.map(message => message.id)).toEqual([localWarning.id, ...truncated.map(row => row.id)])
      expect(refreshed[0]).toBe(localWarning)
      refreshed = reconcileDurableHistory(truncated, refreshed)
    }

    const represented = { ...localWarning, id: 'durable-warning-truncated' }
    refreshed = reconcileDurableHistory([represented, ...truncated], refreshed)
    expect(refreshed.filter(message => message.role === 'system')).toEqual([represented])
  })

  it('does not let a truncated turn claim an identical warning in a later turn', () => {
    const stored = toChatMessages([
      { id: 130, role: 'user', content: 'First prompt.' },
      { id: 131, role: 'assistant', content: 'First answer.' },
      { id: 132, role: 'user', content: 'Second prompt.' },
      { id: 133, role: 'assistant', content: 'Second answer.' }
    ])
    const localWarning = {
      id: 'model-switch-warning-reused',
      role: 'system' as const,
      parts: [{ type: 'text' as const, text: 'warning: Same provider notice.' }]
    }
    const laterWarning = { ...localWarning }
    const representedLater = { ...laterWarning, id: 'durable-warning-later' }
    const previous = [stored[0], localWarning, stored[1], stored[2], laterWarning, stored[3]]
    const next = [stored[1], stored[2], representedLater, stored[3]]

    let refreshed = reconcileDurableHistory(next, previous)
    for (const _ of [1, 2]) {
      expect(refreshed.filter(message => message.role === 'system')).toEqual([localWarning, representedLater])
      expect(refreshed.map(message => message.role)).toEqual(['system', 'assistant', 'user', 'system', 'assistant'])
      refreshed = reconcileDurableHistory(next, refreshed)
    }
  })

  it('keeps an after-answer warning after that answer when the owning user was truncated', () => {
    const stored = toChatMessages([
      { id: 140, role: 'user', content: 'Earlier prompt.' },
      { id: 141, role: 'assistant', content: 'Earlier answer.' },
      { id: 142, role: 'user', content: 'Next prompt.' },
      { id: 143, role: 'assistant', content: 'Next answer.' }
    ])
    const warning = {
      id: 'model-switch-warning-after-answer',
      role: 'system' as const,
      parts: [{ type: 'text' as const, text: 'warning: Provider changed after the reply.' }]
    }
    const previous = [stored[0], stored[1], warning, stored[2], stored[3]]
    const truncated = stored.slice(1)

    const refreshed = reconcileDurableHistory(truncated, previous)
    expect(refreshed.map(row => row.id)).toEqual([stored[1].id, warning.id, stored[2].id, stored[3].id])
    expect(reconcileDurableHistory(truncated, refreshed)).toEqual(refreshed)
  })

  it('keeps an after-answer warning when a later turn stores identical text after two truncating refreshes', () => {
    const stored = toChatMessages([
      { id: 150, role: 'user', content: 'Earlier prompt.' },
      { id: 151, role: 'assistant', content: 'Earlier answer.' },
      { id: 152, role: 'user', content: 'Later prompt.' },
      { id: 153, role: 'assistant', content: 'Later answer.' }
    ])
    const local = {
      id: 'model-switch-warning-earlier',
      role: 'system' as const,
      parts: [{ type: 'text' as const, text: 'warning: Repeated notice.' }]
    }
    const truncated = stored.slice(1)
    const first = reconcileDurableHistory(truncated, [stored[0], stored[1], local, ...stored.slice(2)])
    expect(first.map(row => row.id)).toEqual([stored[1].id, local.id, stored[2].id, stored[3].id])
    const later = { ...local, id: 'stored-warning-later' }
    const second = reconcileDurableHistory([stored[1], stored[2], later, stored[3]], first)
    expect(second.map(row => row.id)).toEqual([stored[1].id, local.id, stored[2].id, later.id, stored[3].id])
    expect(reconcileDurableHistory([stored[1], stored[2], later, stored[3]], second)).toEqual(second)
  })

  it('uses a stable replay occurrence ID while retaining a later identical warning as a new event', () => {
    const warning = { kind: 'model-switch-warning', text: 'Repeated notice.', timestamp: 900 }
    act(() =>
      stream.handleEvent({ type: 'status.update', session_id: SID, replay_epoch: 'epoch-1', seq: 42, payload: warning })
    )
    const first = stream.state(SID).messages.at(-1)!
    act(() =>
      stream.handleEvent({ type: 'status.update', session_id: SID, replay_epoch: 'epoch-1', seq: 42, payload: warning })
    )
    expect(stream.state(SID).messages.filter(row => row.role === 'system')).toEqual([first])
    act(() =>
      stream.handleEvent({ type: 'status.update', session_id: SID, replay_epoch: 'epoch-1', seq: 43, payload: warning })
    )
    expect(stream.state(SID).messages.filter(row => row.role === 'system')).toHaveLength(2)

    act(() =>
      stream.handleEvent({
        type: 'status.update',
        session_id: SID,
        replayEpoch: 'legacy-epoch',
        seq: 44,
        payload: warning
      })
    )
    act(() =>
      stream.handleEvent({
        type: 'status.update',
        session_id: SID,
        replayEpoch: 'legacy-epoch',
        seq: 44,
        payload: warning
      })
    )
    expect(stream.state(SID).messages.filter(row => row.role === 'system')).toHaveLength(3)
  })

  it('keeps a pre-reply warning at its turn boundary after completion and history refresh', () => {
    const stored = toChatMessages([
      { id: 70, role: 'user', content: 'Earlier prompt.' },
      { id: 71, role: 'assistant', content: 'Earlier answer.' },
      { id: 72, role: 'user', content: 'Current prompt.' },
      { id: 73, role: 'assistant', content: 'Current answer.' }
    ])
    // Separate same-millisecond hydrations can mint equal client IDs for different durable users.
    const currentUser = { ...stored[2], id: stored[0].id }
    stream.states.set(SID, { ...stream.state(SID), messages: [...stored.slice(0, 2), currentUser] })

    event('message.start', 700)
    event('status.update', 701, { kind: 'model-switch-warning', text: 'Provider changed before reply.' })
    event('message.delta', 702, { text: 'Current answer.' })
    event('message.complete', 703, { text: 'Current answer.' })

    const local = stream.state(SID).messages
    expect(local.map(message => message.role)).toEqual(['user', 'assistant', 'user', 'system', 'assistant'])
    const warning = local.find(message => message.role === 'system')!
    expect(warning.timestamp).toBe(701)
    let refreshed = reconcileDurableHistory(stored, local)
    expect(refreshed.map(message => message.role)).toEqual(['user', 'assistant', 'user', 'system', 'assistant'])
    expect(refreshed[3]).toBe(warning)
    refreshed = reconcileDurableHistory(stored, refreshed)
    expect(refreshed.map(message => message.role)).toEqual(['user', 'assistant', 'user', 'system', 'assistant'])
    expect(refreshed[3]).toBe(warning)
  })

  it('uses session.info time when it is the only stop boundary', () => {
    event('message.start', 500)
    event('tool.start', 501, { args: {}, name: 'terminal', tool_id: 'call-3' })
    event('session.info', 502.75, { running: false })

    const assistant = stream.state(SID).messages.find(message => message.role === 'assistant')

    expect(assistant?.completedAt).toBe(502.75)
    expect(assistant?.parts[0].completedAt).toBe(502.75)
  })
})
