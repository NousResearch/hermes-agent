// @vitest-environment jsdom
/**
 * The GPT-Live level meter must move with real probe amplitude, not sit at a
 * hardcoded 0.6 while speaking and 0 while silent. The fake session below
 * stands in for VoiceLiveSession and fires the speaking callbacks with the
 * amplitudes the real probe would report.
 */
import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { VoiceLiveHandlers } from '@/lib/voice-live'

vi.mock('@/lib/voice-live', () => {
  interface FakeSession {
    handlers: VoiceLiveHandlers
  }

  const instances: FakeSession[] = []

  class FakeVoiceLiveSession {
    handlers: VoiceLiveHandlers

    constructor(handlers: VoiceLiveHandlers) {
      this.handlers = handlers
      instances.push(this)
    }

    static get instances(): FakeSession[] {
      return instances
    }

    async start(): Promise<void> {
      // No transport in the test double.
    }

    close(): void {
      // Nothing to tear down.
    }

    speak(): void {}

    think(): void {}

    instruct(): void {}

    setMuted(): void {}
  }

  return { VoiceLiveSession: FakeVoiceLiveSession }
})

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      notifications: {
        voice: {
          couldNotStartSession: 'could not start',
          liveDelegationFailed: 'delegation failed',
          liveEnded: 'live ended',
          liveError: 'live error'
        }
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

import { VoiceLiveSession } from '@/lib/voice-live'

import { useVoiceLiveConversation } from './use-voice-live-conversation'

function latestHandlers(): VoiceLiveHandlers {
  const sessions = (VoiceLiveSession as unknown as { instances: Array<{ handlers: VoiceLiveHandlers }> }).instances

  if (sessions.length === 0) {
    throw new Error('expected the hook to construct a VoiceLiveSession')
  }

  return sessions.at(-1)!.handlers
}

describe('GPT-Live voice level meter', () => {
  it('exposes the probe amplitude as level instead of a fixed 0.6/0', async () => {
    const { rerender, result } = renderHook(
      ({ enabled }: { enabled: boolean }) =>
        useVoiceLiveConversation({
          busy: false,
          consumePendingResponse: () => undefined,
          enabled,
          onSubmit: () => undefined,
          pendingResponse: () => null,
          seedHistory: () => []
        }),
      { initialProps: { enabled: false } }
    )

    // The hook starts the session on the false → true enabled transition.
    rerender({ enabled: true })

    // Flush the async start() chain so the fake VoiceLiveSession is constructed.
    await act(async () => {})

    expect(result.current.level).toBe(0)

    // A soft voice and a loud voice must visibly differ on the meter.
    act(() => {
      latestHandlers().onSpeakingChange?.(true, 0.25)
    })
    expect(result.current.level).toBe(0.25)

    act(() => {
      latestHandlers().onSpeakingLevel?.(0.8)
    })
    expect(result.current.level).toBe(0.8)

    act(() => {
      latestHandlers().onSpeakingLevel?.(0.15)
    })
    expect(result.current.level).toBe(0.15)

    act(() => {
      latestHandlers().onSpeakingChange?.(false, 0)
    })
    expect(result.current.level).toBe(0)
  })
})
