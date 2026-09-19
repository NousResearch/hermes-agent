import type { GatewayEvent } from '@hermes/shared'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type ChatMessage, chatMessageText, type GatewayEventPayload } from '@/lib/chat-messages'
import { clearSessionTodos } from '@/store/todos'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

// A turn whose reasoning summary arrives only on the terminal `message.complete`
// frame (the Codex subscription shape) rather than as streamed
// `reasoning.delta`/`reasoning.available` frames must still surface its
// readable summary live — history hydration already renders it after a reload,
// so the live turn must agree with the reloaded one.
const SID = 'session-1'

let stream: MessageStreamHarness

const start = () => act(() => stream.handleEvent({ payload: {}, session_id: SID, type: 'message.start' }))

const delta = (text: string) =>
  act(() => stream.handleEvent({ payload: { text }, session_id: SID, type: 'message.delta' }))

const reasoningDelta = (text: string) =>
  act(() => stream.handleEvent({ payload: { text }, session_id: SID, type: 'reasoning.delta' }))

const complete = (payload: GatewayEventPayload) =>
  act(() => stream.handleEvent({ payload, session_id: SID, type: 'message.complete' }))

function lastAssistant(): ChatMessage | undefined {
  return [...stream.state(SID).messages].reverse().find(message => message.role === 'assistant' && !message.hidden)
}

function reasoningTexts(): string[] {
  return (lastAssistant()?.parts ?? []).filter(part => part.type === 'reasoning').map(part => part.text)
}

describe('useMessageStream completion reasoning fallback', () => {
  beforeEach(() => {
    clearSessionTodos(SID)
  })

  afterEach(() => {
    cleanup()
    clearSessionTodos(SID)
    vi.restoreAllMocks()
  })

  it('appends the completion reasoning when none streamed (settles the streaming bubble)', async () => {
    stream = renderMessageStream(SID)
    await start()
    await delta('done')

    await complete({ reasoning: 'summary text', text: 'done' })

    expect(reasoningTexts()).toEqual(['summary text'])
    expect(chatMessageText(lastAssistant()!)).toBe('done')
  })

  it('appends the completion reasoning when the completion opens a fresh bubble', async () => {
    stream = renderMessageStream(SID)

    await complete({ reasoning: 'summary text', text: 'done' })

    expect(reasoningTexts()).toEqual(['summary text'])
    expect(chatMessageText(lastAssistant()!)).toBe('done')
  })

  it('keeps the streamed reasoning and ignores a differing completion reasoning', async () => {
    stream = renderMessageStream(SID)
    await start()
    await reasoningDelta('streamed reasoning')
    await delta('done')

    await complete({ reasoning: 'completion summary', text: 'done' })

    expect(reasoningTexts()).toEqual(['streamed reasoning'])
  })

  it('ignores missing, empty, whitespace-only, and non-string reasoning', async () => {
    stream = renderMessageStream(SID)
    await start()
    await delta('done')

    await complete({ text: 'done' })
    expect(reasoningTexts()).toEqual([])

    await complete({ reasoning: '', text: 'done' })
    expect(reasoningTexts()).toEqual([])

    await complete({ reasoning: '   \n\t ', text: 'done' })
    expect(reasoningTexts()).toEqual([])

    await act(() =>
      stream.handleEvent({
        payload: { reasoning: 12345, text: 'done' },
        session_id: SID,
        type: 'message.complete'
      } as unknown as GatewayEvent)
    )
    expect(reasoningTexts()).toEqual([])
  })
})
