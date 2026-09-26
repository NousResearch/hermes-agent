import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import {
  $parkedQueueSessions,
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  getQueuedPrompts,
  MAX_AUTO_DRAIN_ATTEMPTS
} from '@/store/composer-queue'
import { clearNotifications } from '@/store/notifications'
import { $sessions, forgetSessionOwnerHintsForSession, setSessionOwnerHint, setSessionsLoading } from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'

import { useBackgroundQueueDrain } from './use-background-queue-drain'

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  $queuedPromptsBySession.set({})
  $parkedQueueSessions.set({})
  $sessions.set([])
  setSessionsLoading(true)
  forgetSessionOwnerHintsForSession('hidden-bot-chat')
  clearNotifications()
  clearAllSessionStates()
})

it.each(['rejection', 'transport error'])(
  'preserves a known hidden session queue after transient %s',
  async failure => {
    vi.useFakeTimers()
    $sessions.set([])
    setSessionsLoading(false)
    clearAllSessionStates()
    setSessionOwnerHint('hidden-bot-chat', { connectionId: 'remote-a', profile: 'bot' })
    const first = enqueueQueuedPrompt('hidden-bot-chat', { text: 'retry this later', attachments: [] })!
    const second = enqueueQueuedPrompt('hidden-bot-chat', { text: 'not attempted yet', attachments: [] })!
    const submitText = vi.fn<(text: string) => Promise<boolean>>()

    if (failure === 'transport error') {
      submitText.mockRejectedValue(new Error('WebSocket temporarily disconnected'))
    } else {
      submitText.mockResolvedValue(false)
    }

    const runtimeMap = { current: new Map([['hidden-bot-chat', 'live-runtime']]) }

    renderHook(() =>
      useBackgroundQueueDrain({
        enabled: true,
        runtimeIdByStoredSessionIdRef: runtimeMap,
        selectedStoredSessionId: 'foreground',
        submitText
      })
    )

    await act(async () => {
      await Promise.resolve()
    })

    for (let attempt = 1; attempt < MAX_AUTO_DRAIN_ATTEMPTS; attempt++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(750)
      })
    }

    expect(submitText).toHaveBeenCalledTimes(MAX_AUTO_DRAIN_ATTEMPTS)
    expect(submitText.mock.calls.every(call => call[0] === first.text)).toBe(true)
    expect(getQueuedPrompts('hidden-bot-chat').map(entry => entry.id)).toEqual([first.id, second.id])
  }
)
