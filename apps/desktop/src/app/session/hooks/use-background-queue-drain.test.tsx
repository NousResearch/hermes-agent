import { act, cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import {
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  flushQueuedPromptMutations,
  getQueuedPrompts
} from '@/store/composer-queue'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'

import { useBackgroundQueueDrain } from './use-background-queue-drain'
import type { SubmitTextOptions, SubmitTextResult } from './use-prompt-actions/utils'

function Harness({
  enabled = true,
  runtimeMap,
  selectedStoredSessionId = 'stored-session-b',
  submitText
}: {
  enabled?: boolean
  runtimeMap: MutableRefObject<Map<string, string>>
  selectedStoredSessionId?: string | null
  submitText: (text: string, options?: SubmitTextOptions) => Promise<SubmitTextResult> | SubmitTextResult
}) {
  useBackgroundQueueDrain({
    enabled,
    runtimeIdByStoredSessionIdRef: runtimeMap,
    selectedStoredSessionId,
    submitText
  })

  return null
}

describe('useBackgroundQueueDrain', () => {
  beforeEach(async () => {
    await flushQueuedPromptMutations()
    vi.useRealTimers()
    window.localStorage.removeItem('hermes.desktop.composerQueue.v1')
    $queuedPromptsBySession.set({})
    clearAllSessionStates()
  })

  afterEach(async () => {
    cleanup()
    await flushQueuedPromptMutations()
    vi.restoreAllMocks()
    vi.useRealTimers()
    window.localStorage.removeItem('hermes.desktop.composerQueue.v1')
    $queuedPromptsBySession.set({})
    clearAllSessionStates()
  })

  it('drains an idle queued prompt for a non-selected background session', async () => {
    const runtimeMap = { current: new Map([['stored-session-a', 'rt-session-a']]) }
    const submitText = vi.fn(async () => true)

    const queued = enqueueQueuedPrompt('stored-session-a', {
      text: 'continue in the background',
      attachments: []
    })

    clearAllSessionStates()

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await waitFor(() => {
      expect(submitText).toHaveBeenCalledWith('continue in the background', {
        attachments: [],
        clientSubmissionId: queued!.id,
        fromQueue: true,
        sessionId: 'rt-session-a',
        storedSessionId: 'stored-session-a'
      })
    })

    await waitFor(() => expect(submitText).toHaveBeenCalledTimes(1))
    expect(getQueuedPrompts('stored-session-a')).toHaveLength(1)
  })

  it('removes exact queue custody immediately for a durable duplicate admission', async () => {
    const runtimeMap = { current: new Map([['stored-session-a', 'rt-session-a']]) }
    const queued = enqueueQueuedPrompt('stored-session-a', { text: 'already admitted', attachments: [] })
    const submitText = vi.fn(async () => ({ accepted: true as const, status: 'duplicate' as const }))

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await waitFor(() => expect(submitText).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(getQueuedPrompts('stored-session-a')).toEqual([]))
    expect(queued).not.toBeNull()
  })

  it('retains a queue lease for queued backend admission', async () => {
    const runtimeMap = { current: new Map([['stored-session-a', 'rt-session-a']]) }
    enqueueQueuedPrompt('stored-session-a', { text: 'queued prompt', attachments: [] })
    const submitText = vi.fn(async () => ({ accepted: true as const, status: 'queued' as const }))

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await waitFor(() => expect(submitText).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(getQueuedPrompts('stored-session-a')[0]?.acceptedAt).toEqual(expect.any(Number)))
  })

  it('leaves the selected session queue to the mounted ChatBar drainer', async () => {
    const runtimeMap = { current: new Map([['stored-session-a', 'rt-session-a']]) }
    const submitText = vi.fn(async () => true)

    enqueueQueuedPrompt('stored-session-a', { text: 'visible queue entry', attachments: [] })
    clearAllSessionStates()

    render(<Harness runtimeMap={runtimeMap} selectedStoredSessionId="stored-session-a" submitText={submitText} />)

    await new Promise(resolve => window.setTimeout(resolve, 0))

    expect(submitText).not.toHaveBeenCalled()
    expect(getQueuedPrompts('stored-session-a')).toHaveLength(1)
  })

  it('does not drain a background session that is still marked working', async () => {
    const runtimeMap = { current: new Map([['stored-session-a', 'rt-session-a']]) }
    const submitText = vi.fn(async () => true)

    enqueueQueuedPrompt('stored-session-a', { text: 'wait for current turn', attachments: [] })
    // Mark the session as working (busy) so the drain should skip it
    publishSessionState('rt-session-a', { ...createClientSessionState('stored-session-a'), busy: true })

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await new Promise(resolve => window.setTimeout(resolve, 0))

    expect(submitText).not.toHaveBeenCalled()
    expect(getQueuedPrompts('stored-session-a')).toHaveLength(1)
  })

  it('passes a null runtime id so submitText can resume stale background sessions by stored id', async () => {
    const runtimeMap = { current: new Map<string, string>() }
    const submitText = vi.fn(async () => true)

    const queued = enqueueQueuedPrompt('stored-session-a', { text: 'resume then send', attachments: [] })

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await waitFor(() => {
      expect(submitText).toHaveBeenCalledWith('resume then send', {
        attachments: [],
        clientSubmissionId: queued!.id,
        fromQueue: true,
        sessionId: null,
        storedSessionId: 'stored-session-a'
      })
    })
  })

  it('claims one queue entry once when two drain owners overlap', async () => {
    const runtimeMap = { current: new Map([['stored-session-a', 'rt-session-a']]) }
    let accept: ((accepted: boolean) => void) | null = null

    const submitText = vi.fn(
      () =>
        new Promise<boolean>(resolve => {
          accept = resolve
        })
    )

    enqueueQueuedPrompt('stored-session-a', { text: 'continue once', attachments: [] })

    render(
      <>
        <Harness runtimeMap={runtimeMap} submitText={submitText} />
        <Harness runtimeMap={runtimeMap} submitText={submitText} />
      </>
    )

    await waitFor(() => expect(submitText).toHaveBeenCalledTimes(1))

    await act(async () => {
      accept?.(true)
      await Promise.resolve()
    })

    await waitFor(() => expect(submitText).toHaveBeenCalledTimes(1))
    expect(getQueuedPrompts('stored-session-a')).toHaveLength(1)
  })

  it('retries a rejected background drain without waiting for another queue or busy-state change', async () => {
    vi.useFakeTimers()

    const runtimeMap = { current: new Map([['stored-session-a', 'rt-session-a']]) }
    const submitText = vi.fn().mockResolvedValueOnce(false).mockResolvedValueOnce(true)

    enqueueQueuedPrompt('stored-session-a', { text: 'retry me', attachments: [] })

    render(<Harness runtimeMap={runtimeMap} submitText={submitText} />)

    await act(async () => {
      await Promise.resolve()
    })

    expect(submitText).toHaveBeenCalledTimes(1)
    expect(getQueuedPrompts('stored-session-a')).toHaveLength(1)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(750)
      await Promise.resolve()
    })

    expect(submitText).toHaveBeenCalledTimes(2)
    expect(getQueuedPrompts('stored-session-a')).toHaveLength(1)
  })
})
