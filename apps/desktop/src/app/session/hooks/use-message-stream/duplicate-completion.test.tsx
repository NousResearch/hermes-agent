import type { GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { appendMidTurnUserMessage } from '@/app/session/hooks/use-prompt-actions/rewind'
import { type ChatMessage, chatMessageText, textPart } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'

import { renderMessageStream } from './test-harness'

const SID = 'duplicate-completion'

function mount() {
  const hydrate = vi.fn(async () => undefined)
  const stream = renderMessageStream(SID, { hydrateFromStoredSession: hydrate })

  const send = (type: GatewayEventName, payload: Record<string, unknown> = {}) =>
    act(() => stream.handleEvent({ type, payload, session_id: SID }))

  return { stream, send, hydrate }
}

afterEach(cleanup)

it('settles identical tool-interim completion once while retaining every completed call', async () => {
  for (const reusedId of [false, true]) {
    const { stream, send, hydrate } = mount()
    await send('message.start')
    await send('message.delta', { text: 'same reply' })

    if (reusedId) {
      await send('tool.start', { name: 'terminal', tool_id: 'call', args: { command: 'pwd' } })
      await send('tool.complete', { name: 'terminal', tool_id: 'call', result: 'first result' })
    }

    await send('message.interim', { text: 'same reply', already_streamed: true })
    const keptId = stream.state().messages.at(-1)!.id
    await send('tool.start', { name: 'terminal', tool_id: 'call', args: { command: 'date' } })
    await send('tool.complete', { name: 'terminal', tool_id: 'call', result: 'second result' })
    await send('message.complete', { text: 'same reply' })

    const messages = stream.state().messages
    const tools = messages.flatMap(message => message.parts).filter(part => part.type === 'tool-call')
    expect(messages).toHaveLength(1)
    expect(messages[0]).toMatchObject({ id: keptId, interim: false, pending: false })
    expect(chatMessageText(messages[0])).toBe('same reply')
    expect(tools.map(tool => tool.result)).toEqual(reusedId ? ['first result', 'second result'] : ['second result'])
    expect(new Set(tools.map(tool => tool.toolCallId)).size).toBe(tools.length)
    expect(stream.state().streamId).toBeNull()
    expect(hydrate).not.toHaveBeenCalled()
    cleanup()
  }
})

it('does not append a final answer already present in a hydrated folded tool card', async () => {
  const folded: ChatMessage = {
    id: 'stored-folded-turn',
    role: 'assistant',
    rowId: 8,
    parts: [
      { type: 'text', text: 'I will inspect the fixture, then give the final result.' },
      { type: 'tool-call', toolCallId: 'call-1', toolName: 'terminal', result: '71' },
      { type: 'text', text: 'The result is 71.' }
    ]
  }

  const user: ChatMessage = { id: 'stored-user', role: 'user', rowId: 7, parts: [textPart('inspect the fixture')] }

  const states = new Map([
    [
      SID,
      {
        ...createClientSessionState(),
        messages: [user, folded],
        interimBoundaryPending: true,
        sawAssistantPayload: true,
        turnLive: true
      }
    ]
  ])

  const hydrate = vi.fn(async () => undefined)
  const stream = renderMessageStream(SID, { hydrateFromStoredSession: hydrate, states })

  await act(() =>
    stream.handleEvent({
      type: 'message.complete',
      session_id: SID,
      payload: { text: 'The result is 71.' }
    })
  )

  const messages = stream.state().messages

  const finalTexts = messages.flatMap(message =>
    message.parts.filter(part => part.type === 'text' && part.text === 'The result is 71.')
  )

  expect(messages).toHaveLength(2)
  expect(finalTexts).toHaveLength(1)
  expect(messages.at(-1)?.id).toBe('stored-folded-turn')
  expect(hydrate).not.toHaveBeenCalled()
})

it('keeps distinct segments, user boundaries, and failures on their own side of completion', async () => {
  const cases = [
    { name: 'distinct interim', interim: 'Checking.' },
    { name: 'longer final', final: 'same reply with more detail' },
    { name: 'distinct live text', live: 'Different streamed text.' },
    { name: 'visible correction', user: true },
    { name: 'hidden directive', user: true, hidden: true },
    { name: 'intervening assistant', assistant: true },
    { name: 'new turn', restart: true },
    { name: 'failed completion', error: 'tool failed' }
  ]

  for (const fixture of cases) {
    const { stream, send } = mount()
    await send('message.start')
    await send('message.interim', { text: fixture.interim ?? 'same reply', already_streamed: true })
    const earlierId = stream.state().messages[0].id

    if (fixture.user) {
      stream.states.set(
        SID,
        appendMidTurnUserMessage(stream.state(), {
          id: 'correction',
          role: 'user',
          parts: [textPart('Use the other file.')],
          hidden: fixture.hidden
        })
      )
    }

    if (fixture.assistant) {
      await send('message.interim', { text: 'Another observation.', already_streamed: true })
    }

    if (fixture.restart) {
      await send('message.start')
    }

    await send('tool.start', { name: 'terminal', tool_id: 'after-boundary', args: { command: 'pwd' } })

    if (fixture.live) {
      await send('message.delta', { text: fixture.live })
    }

    const liveId = stream.state().streamId
    await send('message.complete', {
      text: fixture.final ?? 'same reply',
      ...(fixture.error ? { error: fixture.error, status: 'error' } : {})
    })

    const messages = stream.state().messages
    expect(messages.at(-1)?.id, fixture.name).toBe(liveId)
    expect(chatMessageText(messages.find(message => message.id === earlierId)!), fixture.name).toBe(
      fixture.interim ?? 'same reply'
    )
    expect(
      messages.at(-1)?.parts.some(part => part.type === 'tool-call'),
      fixture.name
    ).toBe(true)

    if (fixture.error) {
      expect(messages.at(-1)?.error).toBe(fixture.error)
    }

    cleanup()
  }
})
