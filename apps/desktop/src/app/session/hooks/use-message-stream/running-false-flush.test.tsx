import type { GatewayEvent } from '@hermes/shared'
// A turn can settle on `session.info running=false` alone (lost or reordered
// message.complete, reconnect gap, provider crash). Text still waiting on the
// flush floor — up to 250 ms while the window is hidden — must land in the
// reply it belongs to, not in a fresh pending bubble seeded after the settle
// cleared the stream id.
import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const SID = 'running-false-flush-session'
let stream: MessageStreamHarness

const emit = (event: GatewayEvent) => act(() => stream.handleEvent(event))

const advance = (ms: number) =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })

const assistantRows = () =>
  (stream.state()?.messages ?? [])
    .filter(message => message.role === 'assistant' && !message.hidden)
    .map(message => ({ pending: Boolean(message.pending), text: chatMessageText(message) }))

async function mountHidden() {
  vi.useFakeTimers()
  vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden')
  stream = renderMessageStream(SID)
  await act(async () => {
    await Promise.resolve()
  })
}

describe('running=false settles queued stream text into its reply', () => {
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('drains the queued tail before the settle while the window is hidden', async () => {
    await mountHidden()

    emit({ payload: { running: true }, session_id: SID, type: 'session.info' })
    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'partial ' }, session_id: SID, type: 'message.delta' })
    await advance(300)

    // The tail is still inside the hidden flush floor when the turn ends.
    emit({ payload: { text: 'tail' }, session_id: SID, type: 'message.delta' })
    await advance(40)
    emit({ payload: { running: false }, session_id: SID, type: 'session.info' })
    await advance(400)

    expect(assistantRows()).toEqual([{ pending: false, text: 'partial tail' }])
  })

  it('does not resurrect a settled turn straggler on an idle running=false (#119543)', async () => {
    await mountHidden()

    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'turn-a' }, session_id: SID, type: 'message.delta' })
    emit({ payload: { text: 'turn-a' }, session_id: SID, type: 'message.complete' })

    // A reordered delta of the settled turn waits in the queue when the idle
    // heartbeat lands; the next turn boundary owns it, not this settle.
    emit({ payload: { text: 'STALE' }, session_id: SID, type: 'message.delta' })
    emit({ payload: { running: false }, session_id: SID, type: 'session.info' })
    emit({ payload: {}, session_id: SID, type: 'message.start' })
    emit({ payload: { text: 'turn-b' }, session_id: SID, type: 'message.delta' })
    await advance(400)
    emit({ payload: {}, session_id: SID, type: 'message.complete' })

    expect(assistantRows().map(row => row.text)).toEqual(['turn-a', 'turn-b'])
  })
})
