import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { overlayConcurrentMessageChanges } from '@/app/session/hooks/use-session-actions/utils'
import { type ChatMessage, chatMessageText } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'

import { renderMessageStream } from './test-harness'
import { STREAM_DELTA_FLUSH_MS } from './utils'

// transcript-integrity "ws drop + reconnect": the re-resume page stores the
// whole reply and the live row adopts it while the stream's tail is still on
// the wire (one delta already queued, one in flight). Those deltas are already
// in the adopted body; the completion that would repair a doubled tail can be
// lost to the same reconnect.
const SID = 'committed-reply-tail'
const TURN = 'turn-a8'
const REPLY = 'A8 survives the drop'

const prompt: ChatMessage = { id: 'user-optimistic', role: 'user', parts: [{ type: 'text', text: 'U8 across a drop' }] }

const page: ChatMessage[] = [
  { id: '16-user', role: 'user', rowId: 16, parts: [{ type: 'text', text: 'U8 across a drop' }] } as ChatMessage,
  { id: '17-assistant', role: 'assistant', rowId: 17, parts: [{ type: 'text', text: REPLY, sourceRowId: 17 }] }
]

describe('a live row that adopted the stored reply', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  it('draws its late tail deltas once when the completion never arrives', async () => {
    vi.useFakeTimers()
    const states = new Map([[SID, { ...createClientSessionState(), messages: [prompt] }]])
    const stream = renderMessageStream(SID, { states })

    const send = (type: string, payload: Record<string, unknown>) =>
      act(() => stream.handleEvent({ payload, session_id: SID, type: type as never }))

    const delta = (text: string, offset: number) => send('message.delta', { offset, segment: 0, text, turn_id: TURN })
    const flush = () => act(async () => void (await vi.advanceTimersByTimeAsync(STREAM_DELTA_FLUSH_MS + 10)))

    await send('message.start', { turn_id: TURN })
    await delta('A8 ', 0)
    await flush()
    const baseline = stream.state().messages
    await delta('survives the', 3)
    await flush()
    await delta(' dr', 15)

    const current = stream.state().messages
    states.set(SID, { ...stream.state(), messages: overlayConcurrentMessageChanges(page, baseline, current) })

    await delta('op', 18)
    await flush()

    const replies = stream.state().messages.filter(message => message.role === 'assistant')

    expect(replies.map(chatMessageText)).toEqual([REPLY])
  })
})
