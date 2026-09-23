import type { GatewayEvent, GatewayEventName } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'
import { clearSessionTodos } from '@/store/todos'

import { renderMessageStream } from './test-harness'

const SID = 'duplicate-interim'

const send = (stream: ReturnType<typeof renderMessageStream>, type: GatewayEventName, payload: Record<string, unknown> = {}) =>
  act(() => stream.handleEvent({ type, payload, session_id: SID } as GatewayEvent))

afterEach(() => {
  cleanup()
  clearSessionTodos(SID)
})

describe('duplicate interim seal re-delivery', () => {
  it('settles a re-delivered message.interim onto the sealed bubble instead of appending a twin', async () => {
    const stream = renderMessageStream(SID)

    await send(stream, 'message.start')
    await send(stream, 'message.delta', { text: 'Today is Wed 23 Sep 2026 — launch is in 8 days.' })
    await send(stream, 'message.interim', { text: 'Today is Wed 23 Sep 2026 — launch is in 8 days.', already_streamed: true })
    // The re-delivered frame — e.g. replayed after a mid-turn WS reconnect.
    await send(stream, 'message.interim', { text: 'Today is Wed 23 Sep 2026 — launch is in 8 days.', already_streamed: true })

    const assistants = stream.state().messages.filter(m => m.role === 'assistant' && !m.hidden)
    const texts = assistants.map(chatMessageText)

    expect(texts.filter(t => t.includes('launch is in 8 days'))).toHaveLength(1)
  })

  it('keeps appending for a genuinely distinct interim after a seal', async () => {
    const stream = renderMessageStream(SID)

    await send(stream, 'message.start')
    await send(stream, 'message.delta', { text: 'Good picture forming.' })
    await send(stream, 'message.interim', { text: 'Good picture forming.', already_streamed: true })
    await send(stream, 'message.delta', { text: 'Let me read three more notes.' })
    await send(stream, 'message.interim', { text: 'Let me read three more notes.', already_streamed: true })

    const texts = stream.state().messages.filter(m => m.role === 'assistant' && !m.hidden).map(chatMessageText)

    expect(texts).toContain('Good picture forming.')
    expect(texts).toContain('Let me read three more notes.')
  })

  it('mirrors the recorded turn (seal → tool group → seal) without doubling narration under re-delivery', async () => {
    const stream = renderMessageStream(SID)

    await send(stream, 'message.start')
    await send(stream, 'tool.start', { name: 'skill_view', tool_id: 'sv1', args: {} })
    await send(stream, 'tool.complete', { name: 'skill_view', tool_id: 'sv1', result: 'ok' })

    await send(stream, 'message.delta', { text: 'Today is Wed 23 Sep 2026 — launch is in 8 days.' })
    await send(stream, 'message.interim', { text: 'Today is Wed 23 Sep 2026 — launch is in 8 days.', already_streamed: true })
    await send(stream, 'message.interim', { text: 'Today is Wed 23 Sep 2026 — launch is in 8 days.', already_streamed: true })

    for (const id of ['rf1', 'rf2', 'rf3', 'rf4', 'rf5']) {
      await send(stream, 'tool.start', { name: 'read_file', tool_id: id, args: {} })
      await send(stream, 'tool.complete', { name: 'read_file', tool_id: id, result: 'content' })
    }

    await send(stream, 'message.delta', { text: 'Good picture forming. Let me read three more notes.' })
    await send(stream, 'message.interim', { text: 'Good picture forming. Let me read three more notes.', already_streamed: true })
    await send(stream, 'message.interim', { text: 'Good picture forming. Let me read three more notes.', already_streamed: true })

    const messages = stream.state().messages.filter(m => m.role === 'assistant' && !m.hidden)
    const texts = messages.map(chatMessageText)

    expect(texts.filter(t => t.includes('launch is in 8 days'))).toHaveLength(1)
    expect(texts.filter(t => t.includes('Good picture forming'))).toHaveLength(1)

    // Tool parts stay keyed by id — exactly one part per call, no matter the churn.
    const tools = messages.flatMap(m => m.parts).filter(p => p.type === 'tool-call')
    expect(new Set(tools.map(t => (t as { toolCallId: string }).toolCallId)).size).toBe(6)
  })

  it('still appends an identical interim authored again in a LATER turn', async () => {
    const stream = renderMessageStream(SID)

    await send(stream, 'message.start')
    await send(stream, 'message.delta', { text: 'same wording' })
    await send(stream, 'message.interim', { text: 'same wording', already_streamed: true })
    await send(stream, 'message.complete', { text: 'done one' })

    // New user turn — the core resets its delivered-interim set per turn, so
    // an identical comment CAN be authored again and must get its own bubble.
    stream.states.set(SID, {
      ...stream.state(SID),
      messages: [
        ...stream.state(SID).messages,
        { id: 'user-2', role: 'user' as const, parts: [{ type: 'text' as const, text: 'again' }] }
      ]
    })
    await send(stream, 'message.start')
    await send(stream, 'message.delta', { text: 'same wording' })
    await send(stream, 'message.interim', { text: 'same wording', already_streamed: true })

    const texts = stream.state().messages.filter(m => m.role === 'assistant' && !m.hidden).map(chatMessageText)
    expect(texts.filter(t => t === 'same wording')).toHaveLength(2)
  })
})
