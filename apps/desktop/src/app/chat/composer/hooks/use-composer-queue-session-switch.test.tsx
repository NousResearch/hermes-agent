import { useStore } from '@nanostores/react'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW } from '@/app/chat/session-view'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $parkedQueueSessions, $queuedPromptsBySession, enqueueQueuedPrompt } from '@/store/composer-queue'
import { $activeSessionId, $busy, $messages, $selectedStoredSessionId, $sessions } from '@/store/session'
import { $sessionStates, publishSessionState } from '@/store/session-states'

import type { QueueEditState } from '../composer-utils'
import type { ChatBarProps } from '../types'

import { useComposerQueue } from './use-composer-queue'

/**
 * End-to-end guard for the session-switch queue drain.
 *
 * The composer's queue is scoped by the STORED id (it comes off the route, so
 * it flips the instant you click another session), while the busy flag it
 * gates on comes from `PRIMARY_SESSION_VIEW.$busy`, keyed by the RUNTIME id
 * that only rebinds once `resumeSession()` lands. While those two disagree the
 * composer is pointed at session A's queue holding session B's busy answer.
 *
 * The auto-drain effect is deliberately level-triggered, not edge-triggered,
 * so a reconnect cannot strand queued entries — which means a single stale
 * `busy === false` render is enough to fire it, and `drainingQueueRef` releases
 * per send, so the WHOLE queue goes, not one entry.
 *
 * `session-view.test.ts` pins the busy oracle itself; this pins the behavior
 * the user actually reported, through the real hook.
 */

const STORED_A = 'stored-session-a'
const RUNTIME_A = 'runtime-a'
const RUNTIME_B = 'runtime-b'

function renderComposerQueueForA() {
  const onSubmit = vi.fn<ChatBarProps['onSubmit']>(async () => true)
  const queueEditRef: { current: QueueEditState | null } = { current: null }

  const hook = renderHook(() => {
    const busy = useStore(PRIMARY_SESSION_VIEW.$busy)

    useComposerQueue({
      activeQueueSessionKey: STORED_A,
      attachments: [],
      busy,
      clearDraft: () => undefined,
      draftRef: { current: '' },
      focusInput: () => undefined,
      loadIntoComposer: () => undefined,
      onCancel: vi.fn(),
      onSteer: undefined,
      onSubmit,
      queueEditRef,
      queueSessionKey: STORED_A,
      sessionId: RUNTIME_A
    })

    return busy
  })

  return { hook, onSubmit }
}

describe('composer queue across a session switch', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $queuedPromptsBySession.set({})
    $parkedQueueSessions.set({})
    $sessionStates.set({})
    $sessions.set([])
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    $messages.set([])
    $busy.set(false)
  })

  afterEach(cleanup)

  const startRunningTurnForA = () => {
    publishSessionState(RUNTIME_A, { ...createClientSessionState(STORED_A), busy: true })
  }

  it('holds the queue through a COLD switch window, where the runtime id is nulled', async () => {
    startRunningTurnForA()
    enqueueQueuedPrompt(STORED_A, { attachments: [], text: 'queued for A #1' })
    enqueueQueuedPrompt(STORED_A, { attachments: [], text: 'queued for A #2' })

    // Mid-switch: the route already selected A, the resume has not bound a
    // runtime id yet.
    act(() => {
      $activeSessionId.set(null)
      $selectedStoredSessionId.set(STORED_A)
    })

    const { hook, onSubmit } = renderComposerQueueForA()

    await act(async () => {
      await Promise.resolve()
    })

    expect(onSubmit).not.toHaveBeenCalled()
    expect(hook.result.current).toBe(true)
    // The authoritative oracle never wavered.
    expect($sessionStates.get()[RUNTIME_A]?.busy).toBe(true)
  })

  it('holds the queue through a WARM switch window, where the pane still holds the old runtime', async () => {
    startRunningTurnForA()
    publishSessionState(RUNTIME_B, { ...createClientSessionState('stored-session-b'), busy: false })
    enqueueQueuedPrompt(STORED_A, { attachments: [], text: 'queued for A #1' })
    enqueueQueuedPrompt(STORED_A, { attachments: [], text: 'queued for A #2' })

    act(() => {
      $activeSessionId.set(RUNTIME_B)
      $selectedStoredSessionId.set(STORED_A)
    })

    const { onSubmit } = renderComposerQueueForA()

    await act(async () => {
      await Promise.resolve()
    })

    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('drains once the turn actually settles', async () => {
    startRunningTurnForA()
    enqueueQueuedPrompt(STORED_A, { attachments: [], text: 'queued for A #1' })

    act(() => {
      $activeSessionId.set(null)
      $selectedStoredSessionId.set(STORED_A)
    })

    const { onSubmit } = renderComposerQueueForA()

    await act(async () => {
      await Promise.resolve()
    })

    expect(onSubmit).not.toHaveBeenCalled()

    act(() => {
      publishSessionState(RUNTIME_A, { ...createClientSessionState(STORED_A), busy: false })
    })

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1))
    expect(onSubmit.mock.calls[0]?.[0]).toBe('queued for A #1')
  })
})
