import { beforeEach, expect, it, vi } from 'vitest'

vi.mock('./onboarding', () => ({ clearFreeTierIntro: vi.fn(), dismissFirstRunOnboarding: vi.fn() }))

// A module reload is the launch boundary: only backend state may resume the guide.
beforeEach(async () => {
  vi.resetModules()
  const { $agentSidebarChoice } = await import('./simplified-agent-mode')
  $agentSidebarChoice.set(true)
})

it('resumes durable setup without inference, a session, or a turn; finish and rerun only write guide state', async () => {
  const run = await import('./setup-guide')
  const { $setupGuidePhase: $onboardingPhase, $onboardingPhase: classicPhase } = await import('./onboarding-phase')
  let state = { run: true, eligible: false, started: true, step: 'model' }

  const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'onboarding.set_run') {
      state = { ...state, ...params }
    } else if (method !== 'onboarding.state') {
      throw new Error(`Unexpected setup side effect: ${method}`)
    }

    return state
  })

  await run.decideOnboarding(request as never)
  expect($onboardingPhase.get()).toBe('shown')
  expect(run.$guideStep.get()).toBe('model')
  const classic = await import('./onboarding-run')
  const { $onboardingSession } = await import('./onboarding-session')
  classicPhase.set('shown')
  $onboardingSession.set({ connectionId: null, runId: 'old-run', runtimeId: 'old-chat', storedId: 'old-chat' })
  classic.noteOnboardingEnded({ outcome: 'done', run_id: 'old-run', stored_session_id: 'old-chat' })
  expect($onboardingPhase.get()).toBe('shown')
  await run.finishOnboarding('done')
  expect(state.run).toBe(false)
  expect($onboardingPhase.get()).toBe('done')
  await run.startOnboarding(request as never)
  expect(state.run).toBe(true)
  expect(run.$guideStep.get()).toBe('preferences')
  const { $agentSidebarChoice } = await import('./simplified-agent-mode')
  const { onboardingSurfaceActive } = await import('./onboarding-presence')
  $agentSidebarChoice.set(false)
  expect(onboardingSurfaceActive()).toBe(false)
  state.step = 'model'
  $agentSidebarChoice.set(true)
  await run.decideOnboarding(request as never, true)
  expect(run.$guideStep.get()).toBe('model')
  expect(request.mock.calls.every(([, params]) => params?.sessionless === true)).toBe(true)
  expect(request.mock.calls.every(([method]) => method === 'onboarding.state' || method === 'onboarding.set_run')).toBe(
    true
  )
})

it('keeps the guide open on a failed skip and ignores a stale peer read after newer progress', async () => {
  const run = await import('./setup-guide')
  const { $setupGuidePhase: $onboardingPhase } = await import('./onboarding-phase')
  let refuse = false
  let deferState = false
  const reply = Promise.withResolvers<unknown>()

  const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'onboarding.state' && deferState) {
      return reply.promise
    }

    if (refuse) {
      throw new Error('Disk is read-only')
    }

    return { run: true, eligible: true, started: true, step: params?.step || 'preferences' }
  })

  await run.startOnboarding(request as never)
  deferState = true
  const stale = run.reconcileOnboarding()
  await run.saveGuideStep('model')
  reply.resolve({ run: false, eligible: true, started: true, step: 'preferences' })
  await stale
  expect(run.$guideStep.get()).toBe('model')
  expect($onboardingPhase.get()).toBe('shown')
  refuse = true
  await expect(run.finishOnboarding('skipped')).rejects.toThrow('Disk is read-only')
  expect($onboardingPhase.get()).toBe('shown')
  refuse = false
  const pendingStep = Promise.withResolvers<unknown>()
  request.mockImplementationOnce(() => pendingStep.promise)
  const stepWrite = run.saveGuideStep('preferences')
  const { $agentSidebarChoice } = await import('./simplified-agent-mode')
  $agentSidebarChoice.set(false)
  pendingStep.resolve({ run: true, eligible: true, started: true, step: 'preferences' })
  await stepWrite
  expect(run.$guideStep.get()).toBe('model')
})

it('does not open a guide when managed policy refuses an explicit restart', async () => {
  const run = await import('./setup-guide')
  const { $setupGuidePhase } = await import('./onboarding-phase')
  const request = vi.fn(async () => ({ run: false, eligible: true, started: false, step: 'preferences' }))

  expect(await run.startOnboarding(request as never)).toBe(false)
  expect($setupGuidePhase.get()).toBe('idle')
  expect(run.$guideRequest.get()).toBeNull()
})
