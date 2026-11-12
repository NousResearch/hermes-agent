import type { OnboardingStartResult } from '@hermes/shared'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { BACKEND_BOOT_WAIT_TIMEOUT_MS } from '@/lib/with-timeout'
import { $chatOnboardingSolo, $introStartExit, $introView } from '@/store/onboarding-intro'
import { $onboardingPhase } from '@/store/onboarding-phase'
import { type OnboardingRequest, startOnboarding } from '@/store/onboarding-run'
import { $activeSessionId, $connection, $selectedStoredSessionId } from '@/store/session'

import { $introCopy } from './intro'
import { firstRunIntro } from './kickoff'

const size = vi.fn()

const started: OnboardingStartResult = {
  resumed: false,
  run_id: 'run-1',
  session_id: 'rt-1',
  stored_session_id: 'st-1'
}

const ensureProfile = async (): Promise<void> => undefined

// SAFETY: the fake answers onboarding.start with its contract result; nothing else reads the answer.
const request = async <T>(): Promise<T> => started as T

// What wiring.tsx's adoptSessionRoute does: the stored chat becomes the selected, routed session.
const adopt = vi.fn(async (storedId: string) => {
  $selectedStoredSessionId.set(storedId)
  $activeSessionId.set('rt-1')
})

beforeEach(() => {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { chatOnboarding: { size } }
  })
  // SAFETY: factsSource reads only `mode`; a local connection skips the machine comparison.
  $connection.set({ mode: 'local' } as never)
  $onboardingPhase.set('idle')
  $introView.set('off')
  $chatOnboardingSolo.set(false)
  $selectedStoredSessionId.set(null)
  size.mockClear()
  adopt.mockClear()
})

describe('first-run kickoff', () => {
  it('lands on the chat onboarding.start returned and opens the intro over it', async () => {
    const views: string[] = []
    const stop = $introView.subscribe(view => views.push(view))

    const opened = await startOnboarding(request, adopt, { ensureProfile, intro: firstRunIntro() })
    stop()

    expect(opened).toBe(true)
    expect(adopt).toHaveBeenCalledWith('st-1', true)
    expect($selectedStoredSessionId.get()).toBe('st-1')
    expect(views).toEqual(['off', 'starting', 'intro'])
    expect($chatOnboardingSolo.get()).toBe(true)
    expect(size).toHaveBeenCalledWith('onboarding')
    // A blank chat plays main's intro copy; the backend's greeting row replaces it.
    expect($introCopy.get()).toBe('playing')
  })

  it('reopens a resumed run without the demo layout, at normal size', async () => {
    // SAFETY: the fake answers onboarding.start with its contract result; nothing else reads the answer.
    const resumed = async <T>(): Promise<T> => ({ ...started, resumed: true }) as T
    const opened = await startOnboarding(resumed, adopt, { ensureProfile, intro: firstRunIntro() })

    expect(opened).toBe(true)
    expect(size).not.toHaveBeenCalled()
    expect($chatOnboardingSolo.get()).toBe(false)
    expect($introView.get()).toBe('ended')
  })

  it('waits on a slow start with no deadline, and the way out skips the run once the start answers', async () => {
    vi.useFakeTimers()
    const calls: [string, unknown, number | undefined][] = []

    let answer: () => void = () => {}

    // A cold start that answers late.
    const slow: OnboardingRequest = async <T>(
      ...[method, params, timeoutMs]: Parameters<OnboardingRequest>
    ): Promise<T> => {
      calls.push([method, params, timeoutMs])

      // SAFETY: the fake answers onboarding.start with its contract result; the skip's answer is unread.
      return method === 'onboarding.start'
        ? new Promise<T>(resolve => (answer = () => resolve(started as T)))
        : ({} as T)
    }

    const opening = startOnboarding(slow, adopt, { ensureProfile, intro: firstRunIntro() })
    await vi.advanceTimersByTimeAsync(BACKEND_BOOT_WAIT_TIMEOUT_MS)
    $introStartExit.get()?.()
    vi.useRealTimers()

    expect($introView.get()).toBe('off')
    expect($onboardingPhase.get()).toBe('skipped')
    answer()

    expect(await opening).toBe(false)
    expect(calls.map(([method, , timeout]) => [method, timeout])).toEqual([
      ['onboarding.start', 0],
      ['onboarding.skip', undefined]
    ])
    expect(calls[1][1]).toEqual({ session_id: 'rt-1' })
    expect(adopt).not.toHaveBeenCalledWith('st-1', true)
  })
})
