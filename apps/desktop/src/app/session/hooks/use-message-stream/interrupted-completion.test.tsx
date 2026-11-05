import { act, cleanup } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { chatMessageText } from '@/lib/chat-messages'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

const cues = vi.hoisted(() => ({
  dispatchNativeNotification: vi.fn(),
  flashPetActivity: vi.fn(),
  markPetUnread: vi.fn(),
  playCompletionSound: vi.fn(),
  setPetActivity: vi.fn()
}))

vi.mock('@/lib/completion-sound', async importOriginal => ({
  ...(await importOriginal<object>()),
  playCompletionSound: cues.playCompletionSound
}))
vi.mock('@/store/native-notifications', async importOriginal => ({
  ...(await importOriginal<object>()),
  dispatchNativeNotification: cues.dispatchNativeNotification
}))
vi.mock('@/store/pet', async importOriginal => ({
  ...(await importOriginal<object>()),
  flashPetActivity: cues.flashPetActivity,
  markPetUnread: cues.markPetUnread,
  setPetActivity: cues.setPetActivity
}))

const SID = 'session-1'

// Whole-message cancellation sentinels older backends settle with. Newer
// backends blank `text` for interrupted turns, so these arrive only from
// legacy gateways — they are metadata, not assistant prose.
const LEGACY_INTERRUPT_SENTINELS = [
  'Operation interrupted.',
  'Operation interrupted: waiting for model response (0.3s elapsed).',
  'Operation interrupted during retry (upstream gateway timeout (504, 42s), attempt 1/3).',
  'Operation interrupted: handling API error (RateLimitError: Too many requests).',
  'Operation interrupted: retrying API call after error (retry 2/3).'
] as const

const PREFIX_COLLIDING_PARTIAL =
  'Operation interrupted: waiting for model response (this phrase describes the log; the real answer continues here.'

let stream: MessageStreamHarness

function mountStream() {
  stream = renderMessageStream(SID)
  // A live turn awaiting its completion frame.
  stream.states.set(SID, { ...stream.state(SID), awaitingResponse: true, busy: true })
}

const completeInterrupted = (text: string) =>
  act(() => stream.handleEvent({ payload: { status: 'interrupted', text }, session_id: SID, type: 'message.complete' }))

const start = () => act(() => stream.handleEvent({ payload: {}, session_id: SID, type: 'message.start' }))

describe('useMessageStream interrupted completion', () => {
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it.each(LEGACY_INTERRUPT_SENTINELS)(
    'settles backend interruption without prose or completion cues, then accepts the queued turn start: %s',
    interruptText => {
      mountStream()
      completeInterrupted(interruptText)

      const settled = stream.state(SID)
      expect(settled.interrupted).toBe(false)
      expect(settled.busy).toBe(false)
      expect(settled.awaitingResponse).toBe(false)
      expect(settled.messages.map(chatMessageText)).not.toContain(interruptText)
      expect(cues.playCompletionSound).not.toHaveBeenCalled()
      expect(cues.flashPetActivity).not.toHaveBeenCalled()
      expect(cues.dispatchNativeNotification).not.toHaveBeenCalled()

      start()

      const restarted = stream.state(SID)
      expect(restarted.busy).toBe(true)
      expect(restarted.awaitingResponse).toBe(true)
      expect(restarted.interrupted).toBe(false)
    }
  )

  it('keeps real partial assistant text that begins with a legacy prefix', () => {
    mountStream()
    completeInterrupted(PREFIX_COLLIDING_PARTIAL)

    expect(stream.state(SID).messages.map(chatMessageText)).toContain(PREFIX_COLLIDING_PARTIAL)
    expect(cues.playCompletionSound).not.toHaveBeenCalled()
    expect(cues.flashPetActivity).not.toHaveBeenCalled()
    expect(cues.dispatchNativeNotification).not.toHaveBeenCalled()
  })
})
