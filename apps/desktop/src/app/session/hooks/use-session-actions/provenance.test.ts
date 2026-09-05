import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

import { type ChatMessage, chatMessageText, toChatMessages } from '@/lib/chat-messages'
import type { SessionMessage, SessionResumeResponse } from '@/types/hermes'

import { appendLiveSessionProjection, chatMessagesEquivalent, overlayConcurrentMessageChanges, preserveLocalPendingTurnMessages } from './utils'

const snapshot = (
  user: string, userOriginated: boolean, startedAt = 10
): Pick<SessionResumeResponse, 'session_id' | 'turn_started_at' | 'inflight'> => ({
  session_id: 'runtime',
  turn_started_at: startedAt,
  inflight: { user, user_originated: userOriginated, assistant: 'partial answer', streaming: true }
})

const withText = (messages: ChatMessage[], text: string) => messages.filter(message => chatMessageText(message) === text)

const todoHeader = '[Your active task list was preserved across context compression]'

describe('backend message provenance', () => {
  it('keeps runtime prose visible without letting it displace an attached live human prompt', () => {
    const prompt = 'current prompt'
    const notice = 'runtime wake with no sentinel'

    const rows = toChatMessages([
      { role: 'user', content: `@image:/tmp/screenshot.png\n${prompt}`, user_originated: true, timestamp: 11 },
      { role: 'assistant', content: 'tool activity', timestamp: 12 },
      { role: 'user', content: notice, display_kind: 'internal_notification', user_originated: false, timestamp: 13 }
    ])

    const restored = appendLiveSessionProjection(rows, snapshot(prompt, true))
    expect(withText(restored, prompt)).toHaveLength(1)
    expect(withText(restored, prompt)[0].attachmentRefs).toEqual(['@image:/tmp/screenshot.png'])
    expect(withText(restored, notice)).toHaveLength(1)
    expect(withText(restored, notice)[0]).toMatchObject({ role: 'user', userOriginated: false })
    expect(restored.slice(0, rows.length)).toEqual(rows)
  })

  it('trusts explicit human origin even when prose resembles a legacy marker', () => {
    const prompt = `${todoHeader} explain this phrase`
    const rows = toChatMessages([{ role: 'user', content: prompt, user_originated: true, timestamp: 11 }])
    const restored = appendLiveSessionProjection(rows, snapshot(prompt, true))

    expect(withText(restored, prompt)).toHaveLength(1)
    expect(withText(restored, prompt)[0].userOriginated).toBe(true)
  })

  it('retains the existing prefix fallback for older gateways', () => {
    const rows = toChatMessages([
      { role: 'user', content: 'current prompt', timestamp: 11 },
      { role: 'user', content: `${todoHeader}\n- current task`, timestamp: 12 }
    ])

    const restored = appendLiveSessionProjection(rows, {
      session_id: 'runtime', turn_started_at: 10,
      inflight: { user: 'current prompt', assistant: 'partial answer', streaming: true }
    })

    expect(rows.every(message => message.userOriginated === undefined)).toBe(true)
    expect(withText(restored, 'current prompt')).toHaveLength(1)
    expect(withText(restored, `${todoHeader}\n- current task`)).toHaveLength(1)
  })

  it('keeps a newly accepted identical human prompt after a completed turn', () => {
    const rows = toChatMessages([
      { role: 'user', content: 'repeat this', user_originated: true, timestamp: 1 },
      { role: 'assistant', content: 'finished answer', timestamp: 2 },
      { role: 'user', content: 'runtime wake', display_kind: 'internal_notification', user_originated: false, timestamp: 3 }
    ])

    const restored = appendLiveSessionProjection(rows, snapshot('repeat this', true))
    expect(withText(restored, 'repeat this')).toHaveLength(2)
    expect(withText(restored, 'runtime wake')).toHaveLength(1)
  })

  it('keeps human input distinct from an identical current runtime notice', () => {
    const rows = toChatMessages([{
      role: 'user', content: 'same text', display_kind: 'internal_notification', user_originated: false, timestamp: 11
    }])

    const restored = appendLiveSessionProjection(rows, snapshot('same text', true))
    expect(withText(restored, 'same text').map(message => message.userOriginated)).toEqual([false, true])
  })

  it.each(['internal_notification', 'auto_continue'])('deduplicates a current %s wake through the same hydration', displayKind => {
    const notice = 'runtime wake'

    const rows = toChatMessages([{
      role: 'user', content: notice, display_kind: displayKind, user_originated: false, timestamp: 11
    }])

    const live = snapshot(notice, false)
    live.inflight = { ...live.inflight, display_kind: displayKind }

    const restored = appendLiveSessionProjection(rows, live)
    const markedRows = rows.map(message => ({ ...message, runtimeTurnStartedAt: live.turn_started_at }))
    expect(restored.filter(message => message.userOriginated === false)).toEqual(markedRows)
    expect(appendLiveSessionProjection(restored, live).filter(message => message.userOriginated === false)).toEqual(markedRows)
  })

  it('reuses a legacy auto-continue timeline event without changing its canonical origin', () => {
    // Legacy auto-continue typing runs after canonical classification, so a
    // persisted untyped row can correctly have true provenance and display as
    // a system event. Its display role already excludes it from human ordinals.
    const rows = toChatMessages([{
      role: 'user', content: 'legacy recovery note', display_kind: 'auto_continue',
      user_originated: true, timestamp: 11
    }])

    const live = snapshot('typed recovery note', false)
    live.inflight = { ...live.inflight, display_kind: 'auto_continue' }

    const restored = appendLiveSessionProjection(rows, live)
    expect(withText(restored, 'resumed interrupted turn')).toHaveLength(1)
    expect(restored[0]).toMatchObject(rows[0])
    expect(rows[0].runtimeTurnStartedAt).toBeUndefined()
    expect(restored[0].userOriginated).toBe(true)
  })

  it('preserves repeated runtime wakes across turns while reconciling a repeated snapshot', () => {
    const rows = toChatMessages([
      { role: 'user', content: 'same wake', display_kind: 'internal_notification', user_originated: false, timestamp: 1 },
      { role: 'assistant', content: 'finished answer', timestamp: 2 }
    ])

    const live = snapshot('same wake', false)
    live.inflight = { ...live.inflight, display_kind: 'internal_notification' }
    const restored = appendLiveSessionProjection(rows, live)

    expect(withText(restored, 'same wake')).toHaveLength(2)
    expect(withText(appendLiveSessionProjection(restored, live), 'same wake')).toHaveLength(2)
  })

  it('does not add a hidden runtime scaffold as an inflight bubble', () => {
    const live = snapshot('hidden scaffold', false)
    live.inflight = { ...live.inflight, display_kind: 'hidden' }
    const restored = appendLiveSessionProjection([], live)

    expect(withText(restored, 'hidden scaffold')).toHaveLength(0)
    expect(withText(restored, 'partial answer')).toHaveLength(1)
  })

  it('reuses a hydrated structured answer when its runtime wake is hidden', () => {
    const rows = toChatMessages([{
      role: 'assistant', content: 'partial answer', reasoning: 'planning', timestamp: 11
    }])

    const live = snapshot('hidden scaffold', false)
    live.inflight = { ...live.inflight, display_kind: 'hidden' }

    const restored = appendLiveSessionProjection(rows, live)
    expect(restored).toMatchObject(rows)
    expect(restored[0].runtimeTurnStartedAt).toBe(live.turn_started_at)
    expect(withText(restored, 'partial answer')).toHaveLength(1)
    expect(restored[0].parts.some(part => part.type === 'reasoning')).toBe(true)
  })

  it('retains newly hydrated provenance when a concurrent reaction replaces the same row', () => {
    const [before] = toChatMessages([{ role: 'user', content: 'notice', timestamp: 11 }])
    const [hydrated] = toChatMessages([{ role: 'user', content: 'notice', user_originated: false, timestamp: 11 }])
    const current: ChatMessage = { ...before, reactions: [{ emoji: '👍', author: 'user', at: 12 }] }

    const merged = overlayConcurrentMessageChanges([hydrated], [before], [current])
    expect(merged[0].userOriginated).toBe(false)
    expect(merged[0].reactions).toEqual(current.reactions)
  })

  it('replaces an unclassified cache entry when provenance arrives without a text change', () => {
    const [before] = toChatMessages([{ role: 'user', content: 'notice', timestamp: 11 }])
    const [after] = toChatMessages([{ role: 'user', content: 'notice', user_originated: false, timestamp: 11 }])

    expect(chatMessagesEquivalent(before, after)).toBe(false)
    expect(chatMessagesEquivalent(after, { ...after })).toBe(true)
  })

  it('uses canonical human ordinals when retaining optimistic prompts on reconnect', () => {
    const rows = toChatMessages([
      { role: 'user', content: 'notice', display_kind: 'internal_notification', user_originated: false, timestamp: 11 },
      { role: 'user', content: 'current prompt', user_originated: true, timestamp: 12 }
    ])

    const optimistic: ChatMessage = {
      id: 'user-optimistic', role: 'user', parts: [{ type: 'text', text: 'current prompt' }]
    }

    const reconciled = preserveLocalPendingTurnMessages(rows, [optimistic])

    expect(withText(reconciled, 'current prompt')).toHaveLength(1)
    expect(withText(reconciled, 'notice')).toHaveLength(1)
  })

  // Generated by test_message_display_provenance.py through real SessionDB,
  // REST, and RPC handlers. Opt in after that test with PROVENANCE_FIXTURE.
  it.skipIf(!process.env.PROVENANCE_FIXTURE)('hydrates and reconciles the real Python transport responses', () => {
    const fixture = JSON.parse(readFileSync(process.env.PROVENANCE_FIXTURE!, 'utf8')) as {
      rest: SessionMessage[]
      rpc: SessionMessage[]
    }

    const rest = toChatMessages(fixture.rest)
    const rpc = toChatMessages(fixture.rpc)

    const signature = (messages: ChatMessage[]) => messages.map(message => ({
      rowId: message.rowId, role: message.role, text: chatMessageText(message), userOriginated: message.userOriginated
    }))

    expect(signature(rest)).toEqual(signature(rpc))

    for (const messages of [rest, rpc]) {
      expect(withText(messages, 'REAL ASK')[0]).toMatchObject({ role: 'user', userOriginated: true })
      expect(messages.filter(message => message.userOriginated === false)).toHaveLength(3)
      expect(withText(messages, 'repeat this')).toHaveLength(2)

      const persisted = appendLiveSessionProjection(messages, snapshot('repeat this', true, 9.5))
      expect(withText(persisted, 'repeat this')).toHaveLength(withText(messages, 'repeat this').length)
      const repeated = appendLiveSessionProjection(messages, snapshot('repeat this', true, 12))
      expect(withText(repeated, 'repeat this')).toHaveLength(withText(messages, 'repeat this').length + 1)

      const wake = messages.find(message => chatMessageText(message) === 'runtime wake with no sentinel')!
      expect(wake).toMatchObject({ role: 'user', userOriginated: false })
      const throughWake = messages.slice(0, messages.indexOf(wake) + 1)
      const live = snapshot(chatMessageText(wake), false, wake.timestamp! - 0.5)
      live.inflight = { ...live.inflight, display_kind: 'internal_notification' }
      const resumed = appendLiveSessionProjection(throughWake, live)
      expect(withText(resumed, chatMessageText(wake))).toHaveLength(1)
    }
  })
})
