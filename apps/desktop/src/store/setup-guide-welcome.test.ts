import { afterEach, beforeEach, expect, it, vi } from 'vitest'

beforeEach(() => {
  vi.resetModules()
  window.localStorage.clear()
})

afterEach(() => {
  window.localStorage.clear()
})

// Dynamic imports intentionally reload the renderer stores; static imports would retain the old launch.
async function renderer(redesign = true) {
  const { $shellChoice } = await import('./shell-variant')
  $shellChoice.set(redesign)

  return {
    guide: await import('./setup-guide'),
    onboarding: await import('./onboarding'),
    phase: (await import('./onboarding-phase')).$setupGuidePhase,
    freeTier: (await import('./free-tier')).$freeTierStatus
  }
}

function backend(run = true) {
  let state = { run, eligible: true, started: false, step: 'preferences' }

  const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    switch (method) {
      case 'onboarding.state':
        return state

      case 'onboarding.set_run':
        state = {
          ...state,
          run: params?.run === true,
          started: state.started || params?.run === true,
          step: typeof params?.step === 'string' ? params.step : state.step
        }

        return state

      case 'setup.status':
        return { provider_configured: true }

      case 'setup.runtime_check':
        return { ok: true, free_tier_route: true }

      case 'free_tier.status':
        return { enabled: true, has_guest: true, notice_pending: true }

      default:
        throw new Error(`Unexpected onboarding side effect: ${method}`)
    }
  })

  return request
}

it.each(['done', 'skipped'] as const)(
  'restores a %s guide before releasing a pending free-model welcome',
  async outcome => {
    const request = backend()
    const first = await renderer()
    await first.guide.decideOnboarding(request as never)
    await first.guide.saveGuideStep('model')
    await first.onboarding.refreshOnboarding({ requestGateway: request as never })
    expect(first.phase.get()).toBe('shown')

    expect(await first.guide.finishOnboarding(outcome)).toBe(true)
    expect(first.onboarding.$desktopOnboarding.get().freeTierReady).toBe(false)

    // A fresh window has no phase or storage latch; only the backend remembers the guide.
    vi.resetModules()
    window.localStorage.clear()
    const fresh = await renderer()
    expect(fresh.phase.get()).toBe('idle')
    await fresh.onboarding.refreshOnboarding({ requestGateway: request as never })
    await fresh.guide.decideOnboarding(request as never)

    expect(fresh.phase.get()).toBe('done')
    expect(fresh.onboarding.$desktopOnboarding.get().freeTierReady).toBe(false)
    expect(fresh.freeTier.get()?.notice_pending).toBe(true)
  }
)

it.each([true, false])(
  'keeps an unrelated free-model introduction when no guide was started (redesign=%s)',
  async redesign => {
    const request = backend(false)
    const fresh = await renderer(redesign)
    await fresh.onboarding.refreshOnboarding({ requestGateway: request as never })
    await fresh.guide.decideOnboarding(request as never)

    expect(fresh.phase.get()).toBe('idle')
    expect(fresh.onboarding.$desktopOnboarding.get().freeTierReady).toBe(true)
    expect(fresh.freeTier.get()?.notice_pending).toBe(true)
  }
)
