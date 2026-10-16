import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { OnboardingRequest } from './onboarding-run'

// The launch's due check runs once per module (`decision ??= …`), so every test loads a fresh copy.
async function freshStores() {
  vi.resetModules()

  const [run, presence, intro, liveSync, session] = await Promise.all([
    import('./onboarding-run'),
    import('./onboarding-presence'),
    import('./onboarding-intro'),
    import('./live-sync'),
    import('./session')
  ])

  // SAFETY: factsSource reads only `mode`; a local connection skips the machine comparison.
  session.$connection.set({ mode: 'local' } as never)

  return { intro, liveSync, presence, run }
}

const ensureProfile = async (): Promise<void> => undefined
const resume = async (): Promise<void> => undefined

/** A requester that keeps the caller's deadline like the real gateway does, and answers
 *  `onboarding.state` only when the test says so. */
function slowBackend() {
  const calls: string[] = []
  let answer: (state: { eligible: boolean; run: boolean }) => void = () => undefined
  const firstState = new Promise<{ eligible: boolean; run: boolean }>(resolve => (answer = resolve))
  let stateReads = 0

  const request: OnboardingRequest = <T>(
    method: string,
    _params?: Parameters<OnboardingRequest>[1],
    timeoutMs?: number
  ): Promise<T> => {
    calls.push(method)

    const reply: Promise<unknown> =
      method === 'onboarding.state' && stateReads++ === 0
        ? firstState.then(state => ({ ...state, run_id: 'run-1', session_id: 'st-1' }))
        : Promise.resolve(
            method === 'onboarding.start'
              ? { resumed: false, run_id: 'run-1', session_id: 'rt-1', stored_session_id: 'st-1' }
              : { eligible: true, run: true, run_id: 'run-1', session_id: 'st-1' }
          )

    const deadline =
      timeoutMs && timeoutMs > 0
        ? new Promise<never>((_resolve, reject) => setTimeout(() => reject(new Error('request timed out')), timeoutMs))
        : null

    // SAFETY: the fake answers each method with its contract result shape.
    return (deadline ? Promise.race([reply, deadline]) : reply) as Promise<T>
  }

  return { answer, calls, request }
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('the launch due check', () => {
  it('waits for a slow answer with no client deadline and still opens onboarding', async () => {
    const { run } = await freshStores()
    const backend = slowBackend()

    const decided = run.decideOnboarding(backend.request, resume, { ensureProfile })
    await vi.advanceTimersByTimeAsync(10_000)
    backend.answer({ eligible: true, run: true })
    await decided

    expect(backend.calls).toContain('onboarding.start')
  })

  it('offers Continue a budget after gateway.ready, and a late due answer after it opens nothing', async () => {
    const { intro, liveSync, presence, run } = await freshStores()
    const backend = slowBackend()

    const decided = run.decideOnboarding(backend.request, resume, { ensureProfile })
    // The socket opened 5 s before the backend said ready; the budget counts from ready.
    await vi.advanceTimersByTimeAsync(5_000)
    liveSync.noteGatewayReady()
    await vi.advanceTimersByTimeAsync(14_000)
    expect(intro.$decideExit.get()).toBeNull()

    await vi.advanceTimersByTimeAsync(1_000)
    intro.$decideExit.get()?.()

    expect(presence.$onboardingDecided.get()).toBe(true)
    expect(intro.$decideExit.get()).toBeNull()

    backend.answer({ eligible: true, run: true })
    await decided

    expect(backend.calls).toEqual(['onboarding.state'])
  })
})
