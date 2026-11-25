import { afterEach, describe, expect, it, vi } from 'vitest'

import { enSandbox } from '@/i18n/en_sandbox'

import {
  $backendTransitions,
  BACKEND_TRANSITION_MAX_MS,
  backendTransitionFor,
  backendTransitionLabel,
  beginBackendTransition,
  endBackendTransition,
  resetBackendTransitionsForTests,
  waitForBackendTransitionEnd
} from './backend-transition'

afterEach(() => {
  vi.useRealTimers()
  resetBackendTransitionsForTests()
})

describe('planned backend transitions', () => {
  it('a waiter resolves true when the transition ends as planned and false when it fails', async () => {
    beginBackendTransition('work', 'restarting', { source: 'launcher' })
    const planned = waitForBackendTransitionEnd('work', 10_000)

    endBackendTransition('work', { source: 'launcher' })
    await expect(planned).resolves.toBe(true)

    beginBackendTransition('work', 'restarting', { source: 'launcher' })
    const failed = waitForBackendTransitionEnd('work', 10_000)

    endBackendTransition('work', { ok: false, source: 'launcher' })
    await expect(failed).resolves.toBe(false)
    await expect(waitForBackendTransitionEnd('work', 10)).resolves.toBe(true)
  })

  it('only its source ends it, and its deadline ends it as failed', async () => {
    vi.useFakeTimers()
    beginBackendTransition(null, 'stopping', { source: 'switch' })
    endBackendTransition('default', { source: 'launcher' })
    expect(backendTransitionFor('default')?.kind).toBe('stopping')

    const waiting = waitForBackendTransitionEnd('default', BACKEND_TRANSITION_MAX_MS * 2)

    await vi.advanceTimersByTimeAsync(BACKEND_TRANSITION_MAX_MS)
    await expect(waiting).resolves.toBe(false)
    expect($backendTransitions.get()).toEqual({})
  })

  it('a waiter gives up after its own timeout', async () => {
    vi.useFakeTimers()
    beginBackendTransition('work', 'restarting', { source: 'launcher' })
    const waiting = waitForBackendTransitionEnd('work', 5_000)

    await vi.advanceTimersByTimeAsync(5_000)
    await expect(waiting).resolves.toBe(false)
    expect(backendTransitionFor('work')).not.toBeNull()
  })

  it('names the state', () => {
    for (const kind of ['starting', 'restarting', 'stopping'] as const) {
      beginBackendTransition('work', kind, { source: 'switch' })
      expect(backendTransitionLabel(backendTransitionFor('work')!, enSandbox)).toBe(enSandbox[kind])
    }

    expect(enSandbox).toEqual({
      restarting: 'Restarting sandbox…',
      starting: 'Starting sandbox…',
      stopping: 'Stopping sandbox…'
    })
  })

  it('keeps its start time when the same source updates it', () => {
    vi.useFakeTimers()
    vi.setSystemTime(1_000)
    beginBackendTransition('work', 'starting', { source: 'launcher' })
    vi.setSystemTime(5_000)
    beginBackendTransition('work', 'restarting', { source: 'launcher' })

    expect(backendTransitionFor('work')).toEqual({ kind: 'restarting', profile: 'work', since: 1_000, source: 'launcher' })
  })

  it.each([
    { label: 'another kind', next: { kind: 'stopping', source: 'switch' } },
    { label: 'another source', next: { kind: 'starting', source: 'launcher' } }
  ] as const)('a transition of $label that replaces one gets a deadline of its own', async ({ next }) => {
    vi.useFakeTimers()
    beginBackendTransition('work', 'starting', { source: 'switch' })

    // Ten seconds before the first one's deadline another begins: a stop right after a slow start.
    await vi.advanceTimersByTimeAsync(BACKEND_TRANSITION_MAX_MS - 10_000)
    beginBackendTransition('work', next.kind, { source: next.source })

    await vi.advanceTimersByTimeAsync(20_000)
    expect(backendTransitionFor('work')?.kind).toBe(next.kind)

    await vi.advanceTimersByTimeAsync(BACKEND_TRANSITION_MAX_MS)
    expect(backendTransitionFor('work')).toBeNull()
  })
})
