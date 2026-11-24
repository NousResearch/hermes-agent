// A chat's queue that is not on screen drains in the background (use-background-queue-drain.ts).
// While its profile's backend is away on purpose (store/backend-transition.ts: a restart holds a
// message in the queue), the queue waits there too, and other profiles' queues keep draining.
import { act, cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import {
  beginBackendTransition,
  endBackendTransition,
  resetBackendTransitionsForTests
} from '@/store/backend-transition'
import {
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  resetFrozenQueuedTransportsForTests
} from '@/store/composer-queue'
import { $sessions, setSessions, setSessionsLoading } from '@/store/session'
import { clearAllSessionStates } from '@/store/session-states'
import { makeSessionInfo } from '@/test/session-info'

import { useBackgroundQueueDrain } from './use-background-queue-drain'
import type { SubmitTextOptions } from './use-prompt-actions/utils'

type SubmitText = (text: string, options?: SubmitTextOptions) => Promise<boolean>

function Harness({ submitText }: { submitText: SubmitText }) {
  useBackgroundQueueDrain({
    enabled: true,
    runtimeIdByStoredSessionIdRef: { current: new Map() },
    selectedStoredSessionId: 'stored-on-screen',
    submitText
  })

  return null
}

const settle = (ms = 0) => act(() => new Promise<void>(resolve => window.setTimeout(resolve, ms)))

beforeEach(() => {
  window.localStorage.removeItem('hermes.desktop.composerQueue.v1')
  resetFrozenQueuedTransportsForTests()
  clearAllSessionStates()
  setSessions([
    makeSessionInfo({ id: 'stored-work', profile: 'work' }),
    makeSessionInfo({ id: 'stored-default', profile: 'default' })
  ])
  setSessionsLoading(false)
})

afterEach(() => {
  cleanup()
  resetBackendTransitionsForTests()
  $queuedPromptsBySession.set({})
  resetFrozenQueuedTransportsForTests()
  $sessions.set([])
  setSessionsLoading(true)
  clearAllSessionStates()
})

it("holds a background chat's queue while its profile's backend restarts, then drains it", async () => {
  beginBackendTransition('work', 'restarting', { source: 'launcher' })
  enqueueQueuedPrompt('stored-work', { attachments: [], text: 'held for the restart' })
  const submitText = vi.fn<SubmitText>(async () => true)

  render(<Harness submitText={submitText} />)
  await settle(100)

  expect(submitText).not.toHaveBeenCalled()

  act(() => endBackendTransition('work', { source: 'launcher' }))

  await waitFor(() => expect(submitText).toHaveBeenCalledTimes(1))
  expect(submitText.mock.calls[0]?.[1]).toMatchObject({ fromQueue: true, storedSessionId: 'stored-work' })
})

it("keeps draining another profile's background queue meanwhile", async () => {
  beginBackendTransition('work', 'restarting', { source: 'launcher' })
  enqueueQueuedPrompt('stored-default', { attachments: [], text: 'not affected' })
  const submitText = vi.fn<SubmitText>(async () => true)

  render(<Harness submitText={submitText} />)

  await waitFor(() => expect(submitText).toHaveBeenCalledTimes(1))
  expect(submitText.mock.calls[0]?.[1]).toMatchObject({ storedSessionId: 'stored-default' })
})
