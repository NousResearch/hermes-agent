import type { OnboardingStartParams } from '@hermes/shared'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { setPrimaryGatewayConnectionId } from './gateway'
import { $onboardingPhase } from './onboarding-phase'
import { $onboardingDecided, $onboardingSurfaceClear } from './onboarding-presence'
import {
  $onboardingEnded,
  $onboardingSettleFrom,
  decideOnboarding,
  noteOnboardingEnded,
  type OnboardingRequest,
  reconcileOnboarding,
  skipOnboarding,
  startOnboarding
} from './onboarding-run'
import { isOnboardingChat } from './onboarding-session'
import { $activeGatewayProfile, $freshSessionRequest, $newChatProfile } from './profile'
import { $connection, $gatewayState, $selectedStoredSessionId } from './session'
import { $sessionTiles } from './session-states'

const ensureProfile = async (): Promise<void> => undefined

// What resumeSession does for the test: the stored chat becomes the one on screen.
const resume = vi.fn(async (storedId: string) => $selectedStoredSessionId.set(storedId))

// `runId`: the run onboarding.state reports open; null once something ended it.
function backend(state: { eligible: boolean; run: boolean }, runId: null | string = 'run-1') {
  const calls: string[] = []
  const starts: Parameters<OnboardingRequest>[1][] = []
  const started = { resumed: false, run_id: 'run-1', session_id: 'rt-1', stored_session_id: 'st-1' }
  const read = { ...state, run_id: runId, session_id: runId && 'st-1' }

  const request = async <T>(method: string, params?: Parameters<OnboardingRequest>[1]): Promise<T> => {
    calls.push(method)

    if (method === 'onboarding.start') {
      starts.push(params)
    }

    // SAFETY: the fake answers onboarding.state and onboarding.start with their contract result shapes.
    return (method === 'onboarding.state' ? read : started) as T
  }

  return { calls, request, starts }
}

beforeEach(() => {
  // SAFETY: factsSource reads only `mode`; a local connection skips the machine comparison.
  $connection.set({ mode: 'local' } as never)
  setPrimaryGatewayConnectionId('local')
  $onboardingPhase.set('idle')
  $onboardingEnded.set(null)
  $onboardingSettleFrom.set(null)
  $selectedStoredSessionId.set(null)
})

describe('onboarding run', () => {
  it('opens the onboarding chat when due and never shows a clear moment before it', async () => {
    const clear: boolean[] = []
    const stop = $onboardingSurfaceClear.subscribe(value => clear.push(value))
    const { calls, request, starts } = backend({ eligible: true, run: true })

    await decideOnboarding(request, resume, { ensureProfile })
    stop()

    // The last read checks for an end sent while the chat was opening.
    expect(calls).toEqual(['onboarding.state', 'onboarding.start', 'onboarding.state'])
    // A new launch: a run the app was closed on does not count, so the backend starts over.
    expect(starts).toEqual([expect.objectContaining({ fresh_launch: true })])
    expect(resume).toHaveBeenCalledWith('st-1', true)
    expect($onboardingDecided.get()).toBe(true)
    expect(clear).not.toContain(true)
  })

  it('keeps the run on a reconnect in the same launch: it is re-read, never started again', async () => {
    const { calls, request, starts } = backend({ eligible: true, run: true })
    await startOnboarding(request, resume, { ensureProfile })
    calls.length = 0

    $gatewayState.set('closed')
    $gatewayState.set('open')
    // The wiring effect runs the launch decision again on every open socket; it answered once already.
    await decideOnboarding(request, resume, { ensureProfile })

    await vi.waitFor(() => expect(calls).toEqual(['onboarding.state']))
    expect(starts).toEqual([expect.not.objectContaining({ fresh_launch: true })])
    expect($onboardingPhase.get()).toBe('shown')
  })

  it('ends once when the backend reports the run done, and frees the first-run surfaces', async () => {
    const { request } = backend({ eligible: true, run: true })

    await startOnboarding(request, resume, { ensureProfile })
    noteOnboardingEnded({ outcome: 'done', run_id: 'run-1', stored_session_id: 'st-1' })
    noteOnboardingEnded({ outcome: 'skipped', run_id: 'run-1', stored_session_id: 'st-1' })

    expect($onboardingPhase.get()).toBe('done')
    expect($onboardingEnded.get()).toBe('done')
    expect($onboardingSurfaceClear.get()).toBe(true)
  })

  it('frees the first-run surfaces in another chat mid-run and arms the end-of-onboarding wait', async () => {
    const { request } = backend({ eligible: true, run: true })

    await startOnboarding(request, resume, { ensureProfile })
    expect($onboardingSurfaceClear.get()).toBe(false)

    $selectedStoredSessionId.set('another-chat')

    expect($onboardingPhase.get()).toBe('shown')
    expect($onboardingSurfaceClear.get()).toBe(true)
    expect($onboardingSettleFrom.get()).not.toBeNull()
  })

  it('ignores the end of another run, like the old run a rerun closed', async () => {
    const { request } = backend({ eligible: true, run: true })
    await startOnboarding(request, resume, { ensureProfile })

    noteOnboardingEnded({ outcome: 'replaced', run_id: 'run-old', stored_session_id: 'st-old' })

    expect($onboardingPhase.get()).toBe('shown')
    expect($onboardingEnded.get()).toBeNull()
  })

  it('knows the onboarding chat under the new runtime id a reconnect gives it', async () => {
    const { request } = backend({ eligible: true, run: true })
    await startOnboarding(request, resume, { ensureProfile })
    // SAFETY: storedSessionIdForRuntimeId reads only these two tile fields.
    $sessionTiles.set([{ runtimeId: 'rt-after-restart', storedSessionId: 'st-1' } as never])

    expect(isOnboardingChat('rt-after-restart')).toBe(true)
    expect(isOnboardingChat('rt-other')).toBe(false)
    $sessionTiles.set([])
  })

  it('ends a run that ended while its chat was opening', async () => {
    const { request } = backend({ eligible: true, run: true }, null)

    await startOnboarding(request, resume, { ensureProfile })
    await vi.waitFor(() => expect($onboardingEnded.get()).toBe('skipped'))
  })

  it('ends a run the backend no longer has open when it re-reads the state', async () => {
    const { request } = backend({ eligible: true, run: true })
    await startOnboarding(request, resume, { ensureProfile })

    // SAFETY: the fakes answer onboarding.state with its contract result shape.
    const open = async <T>(): Promise<T> => ({ eligible: true, run: true, run_id: 'run-1', session_id: 'st-1' }) as T
    // SAFETY: as above.
    const reread = async <T>(): Promise<T> => ({ eligible: true, run: false, run_id: null, session_id: null }) as T

    await reconcileOnboarding(open)
    expect($onboardingPhase.get()).toBe('shown')
    await reconcileOnboarding(reread)
    expect($onboardingEnded.get()).toBe('skipped')
  })

  it('keeps the run while the window reads another computer, and re-reads on its own', async () => {
    const { request } = backend({ eligible: true, run: true })
    await startOnboarding(request, resume, { ensureProfile })

    // SAFETY: the fake answers onboarding.state with its contract result shape.
    const noRun = async <T>(): Promise<T> => ({ eligible: true, run: false, run_id: null, session_id: null }) as T

    setPrimaryGatewayConnectionId('homelab')
    await reconcileOnboarding(noRun)
    expect($onboardingPhase.get()).toBe('shown')

    setPrimaryGatewayConnectionId('local')
    await reconcileOnboarding(noRun)
    expect($onboardingEnded.get()).toBe('skipped')
  })

  it('stays on the default profile when the start fails', async () => {
    $activeGatewayProfile.set('work')
    const switched = vi.fn(async (_profile: string) => undefined)
    const refuse = async (): Promise<never> => Promise.reject(new Error('backend down'))

    expect(await startOnboarding(refuse, resume, { ensureProfile: switched })).toBe(false)
    expect(switched.mock.calls).toEqual([['default']])
    $activeGatewayProfile.set('default')
  })

  it('opens a new chat in the default profile on Skip setup', async () => {
    $activeGatewayProfile.set('work')
    const { request } = backend({ eligible: true, run: true })
    await startOnboarding(request, resume, { ensureProfile })
    const fresh = $freshSessionRequest.get()

    await skipOnboarding(request, 'rt-1')
    $activeGatewayProfile.set('default')

    expect($onboardingEnded.get()).toBe('skipped')
    expect($freshSessionRequest.get()).toBe(fresh + 1)
    expect($newChatProfile.get()).toBe('default')
  })

  it('acks the free-tier notice when the onboarding intro opens', async () => {
    const { calls, request } = backend({ eligible: true, run: true })
    const intro = { failed: vi.fn(), opened: vi.fn(), started: vi.fn(), starting: vi.fn() }

    await startOnboarding(request, resume, { ensureProfile, intro })

    expect(intro.opened).toHaveBeenCalled()
    expect(calls).toContain('free_tier.ack_notice')
  })

  it('releases the boot hold when the user takes the way out while the start is still pending', async () => {
    let exit: () => void = () => undefined

    const intro = {
      failed: vi.fn(),
      opened: vi.fn(),
      started: vi.fn(),
      starting: (leave: () => void) => (exit = leave)
    }

    const request = (method: string) =>
      method === 'onboarding.start' ? new Promise<never>(() => undefined) : Promise.reject(new Error(method))

    $onboardingDecided.set(false)

    // `starting` runs before the first await, so the way out exists as soon as the start is called.
    void startOnboarding(request, resume, { ensureProfile, intro })
    exit()

    expect($onboardingDecided.get()).toBe(true)
  })

  it("keeps a rerun going when the old run's withdrawn card reports skipped_all first", async () => {
    const { request } = backend({ eligible: true, run: true })
    await startOnboarding(request, resume, { ensureProfile })

    // The backend withdraws the old card while the rerun's start is still answering.
    const rerun = async <T>(method: string): Promise<T> => {
      if (method === 'onboarding.start') {
        noteOnboardingEnded({ outcome: 'replaced', run_id: 'run-1', stored_session_id: 'st-1' })
      }

      // SAFETY: the fake answers onboarding.start with its contract result shape.
      return { resumed: false, run_id: 'run-2', session_id: 'rt-2', stored_session_id: 'st-2' } as T
    }

    await startOnboarding(rerun, resume, { ensureProfile, rerun: true })

    expect($onboardingPhase.get()).toBe('shown')
    expect($onboardingEnded.get()).toBeNull()
  })

  it('asks for a fresh-install run only on a --fresh-install launch, and never on a rerun', async () => {
    const sent: boolean[] = []

    const request = async <T>(method: string, params?: OnboardingStartParams): Promise<T> => {
      if (method === 'onboarding.start') {
        sent.push(params !== undefined && 'assume_fresh' in params)
      }

      // SAFETY: the fake answers onboarding.start with its contract result shape.
      return { resumed: false, run_id: 'run-1', session_id: 'rt-1', stored_session_id: 'st-1' } as T
    }

    const launch = (assumeFreshInstall: boolean): void => {
      Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { assumeFreshInstall } })
    }

    try {
      launch(false)
      await startOnboarding(request, resume, { ensureProfile })
      launch(true)
      await startOnboarding(request, resume, { ensureProfile })
      await startOnboarding(request, resume, { ensureProfile, rerun: true })
    } finally {
      Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: undefined })
    }

    // Only the fresh-install launch carries the field at all.
    expect(sent).toEqual([false, true, false])
  })

  it('keeps the run when the backend refuses the skip', async () => {
    const { request } = backend({ eligible: true, run: true })
    await startOnboarding(request, resume, { ensureProfile })

    const refuse = async (): Promise<never> => Promise.reject(new Error('skip refused'))

    await expect(skipOnboarding(refuse, 'rt-1')).rejects.toThrow('skip refused')
    expect($onboardingPhase.get()).toBe('shown')
    expect($onboardingEnded.get()).toBeNull()
  })
})
