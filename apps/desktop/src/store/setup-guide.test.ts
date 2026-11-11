import { beforeEach, expect, it, vi } from 'vitest'

vi.mock('./onboarding', () => ({ clearFreeTierIntro: vi.fn(), dismissFirstRunOnboarding: vi.fn() }))

// A module reload is the launch boundary: only backend state may resume the guide.
beforeEach(async () => {
  vi.resetModules()
  const { $shellChoice } = await import('./shell-variant')
  $shellChoice.set(true)
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
  const { $shellChoice } = await import('./shell-variant')
  const { onboardingSurfaceActive } = await import('./onboarding-presence')
  $shellChoice.set(false)
  expect(onboardingSurfaceActive()).toBe(false)
  state.step = 'model'
  $shellChoice.set(true)
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
  const { $shellChoice } = await import('./shell-variant')
  $shellChoice.set(false)
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

it('never writes a stale automatic-resume snapshot over a peer window', async () => {
  const run = await import('./setup-guide')
  const snapshot = Promise.withResolvers<unknown>()
  let durable = { run: true, eligible: false, started: true, step: 'preferences' }
  const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'onboarding.state') {
      return snapshot.promise
    }
    durable = { ...durable, ...params }
    return durable
  })

  const deciding = run.decideOnboarding(request as never)
  const stale = { ...durable }
  durable = { ...durable, step: 'model' }
  snapshot.resolve(stale)
  await deciding

  expect(durable.step).toBe('model')
  expect(request.mock.calls.map(([method]) => method)).toEqual(['onboarding.state'])
  request.mockImplementation(async () => durable)
  await run.reconcileOnboarding()
  expect(run.$guideStep.get()).toBe('model')
})

it.each([
  ['old schema', 'Invalid params: sessionless Extra inputs are not permitted [type=extra_forbidden]', true],
  ['version skew', 'App and backend out of sync (different versions)', true],
  ['network failure', 'Gateway disconnected', false]
])('offers explicit recovery for %s without falling back to a setup chat', async (_name, message, upgradeRequired) => {
  const run = await import('./setup-guide')
  const { $setupGuidePhase } = await import('./onboarding-phase')
  const request = vi.fn().mockRejectedValue(new Error(message))

  await run.decideOnboarding(request as never)
  expect(run.$guideRecovery.get()).toMatchObject({ message, upgradeRequired })
  expect($setupGuidePhase.get()).toBe('idle')
  expect(request).toHaveBeenCalledOnce()
  expect(request).toHaveBeenCalledWith('onboarding.state', { sessionless: true }, 0, undefined)

  request.mockResolvedValue({ run: true, eligible: false, started: true, step: 'model' })
  await run.$guideRecovery.get()!.retry()
  expect(run.$guideRecovery.get()).toBeNull()
  expect($setupGuidePhase.get()).toBe('shown')
  expect(request.mock.calls.every(([method]) => method === 'onboarding.state')).toBe(true)
})

it('offers recovery for Run again and keeps classic mode free of guide recovery', async () => {
  const run = await import('./setup-guide')
  const { $shellChoice } = await import('./shell-variant')
  const request = vi.fn().mockRejectedValue(new Error('sessionless: extra_forbidden'))
  expect(await run.startOnboarding(request as never)).toBe(false)
  expect(run.$guideRecovery.get()?.upgradeRequired).toBe(true)

  run.$guideRecovery.set(null)
  $shellChoice.set(false)
  await run.decideOnboarding(request as never)
  expect(run.$guideRecovery.get()).toBeNull()
})

it('uses the atomic fresh latch and displays progress advanced by a peer since the snapshot', async () => {
  const run = await import('./setup-guide')
  const snapshot = Promise.withResolvers<unknown>()
  const request = vi.fn(async (method: string) => {
    if (method === 'onboarding.state') {
      return snapshot.promise
    }
    // The atomic latch returns the peer's current state, not the stale read.
    return { run: true, eligible: true, started: true, step: 'model' }
  })
  const deciding = run.decideOnboarding(request as never)
  snapshot.resolve({ run: true, eligible: true, started: false, step: 'preferences' })
  await deciding
  expect(request).toHaveBeenCalledWith('onboarding.set_run', { run: true, sessionless: true }, undefined, undefined)
  expect(run.$guideStep.get()).toBe('model')
})

it('pins recovery retries to the failed gateway and profile, not the new foreground owner', async () => {
  const gateway = await import('./gateway')
  const api = await import('@/api/client')
  vi.spyOn(gateway, 'activeGateway').mockReturnValue({} as never)
  const routed = vi.spyOn(gateway, 'requestGatewayForAgent')
    .mockRejectedValueOnce(new Error('sessionless: extra_forbidden'))
    .mockResolvedValue({ run: true, eligible: false, started: true, step: 'model' })
  api.setApiRequestConnection('alpha')
  api.setApiRequestProfile('research')

  try {
    const run = await import('./setup-guide')
    await run.decideOnboarding(vi.fn() as never)
    api.setApiRequestConnection('beta')
    api.setApiRequestProfile('default')
    await run.$guideRecovery.get()!.retry()
    expect(routed.mock.calls.map(call => call.slice(0, 3))).toEqual([
      ['alpha', 'research', 'onboarding.state'],
      ['alpha', 'research', 'onboarding.state']
    ])
    expect(run.$guideScope.get()).toEqual({ connectionId: 'alpha', profile: 'research' })
  } finally {
    vi.restoreAllMocks()
    api.setApiRequestConnection(null)
    api.setApiRequestProfile(null)
  }
})
