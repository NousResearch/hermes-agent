import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $localModelsEnabled } from '@/store/local-models-flag'
import * as notifications from '@/store/notifications'
import { markOnboardingDecided } from '@/store/onboarding-presence'

import { $desktopOnboarding, type DesktopOnboardingState, type OnboardingContext, refreshOnboarding } from './onboarding'

markOnboardingDecided()

const SERVER_OFF =
  'The local model server is turned off. Start it from the Hermes desktop app or `hermes model` → Local models, or switch to another model with /model.'

function coldState(): DesktopOnboardingState {
  return {
    configured: null,
    flow: { status: 'idle' },
    mode: 'oauth',
    providers: null,
    reason: null,
    requested: false,
    firstRunSkipped: false,
    manual: false,
    localEndpoint: false,
    freeTierReady: false
  }
}

function localModelGateway(runtimeCheck: Record<string, unknown>): OnboardingContext['requestGateway'] {
  return async method => {
    if (method === 'setup.status') {
      return { provider_configured: true } as never
    }

    if (method === 'setup.runtime_check') {
      return runtimeCheck as never
    }

    throw new Error(`unexpected gateway method: ${method}`)
  }
}

describe('refreshOnboarding with the local model server down', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $desktopOnboarding.set(coldState())
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api: async () => ({ providers: [] }) }
    })
  })

  afterEach(() => {
    window.localStorage.clear()
    $desktopOnboarding.set(coldState())
    $localModelsEnabled.set(false)
    vi.restoreAllMocks()
  })

  it('keeps the app open and names the server, never provider sign-in', async () => {
    $localModelsEnabled.set(true)
    const notify = vi.spyOn(notifications, 'notify')

    const ready = await refreshOnboarding({
      requestGateway: localModelGateway({ error: SERVER_OFF, error_code: 'local_runtime_unavailable', ok: false })
    })

    const state = $desktopOnboarding.get()
    expect(ready).toBe(false)
    // The onboarding overlay opens only on configured === false.
    expect(state.configured).toBe(true)
    expect(state.reason).toBe(SERVER_OFF)
    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({
        action: expect.objectContaining({ label: 'Open Local models' }),
        id: 'local-runtime-unavailable',
        // Desktop has the pane, so its notice names it beside the button that opens it.
        message: expect.stringContaining('Settings → Providers → Local models')
      })
    )
  })

  it('offers another model where there is no Local models pane (the Webapp)', async () => {
    const notify = vi.spyOn(notifications, 'notify')

    await refreshOnboarding({
      requestGateway: localModelGateway({ error: SERVER_OFF, error_code: 'local_runtime_unavailable', ok: false })
    })

    expect($desktopOnboarding.get().configured).toBe(true)
    const shown = notify.mock.calls.at(-1)?.[0]
    expect(shown?.action?.label).toBe('Choose another model')
    expect(shown?.message).toContain('Hermes desktop app')
    expect(shown?.message).not.toContain('Local models')
  })

  it('still opens onboarding for an uncoded failure (a missing credential)', async () => {
    await refreshOnboarding({
      requestGateway: localModelGateway({ error: 'No usable credentials found for openrouter.', ok: false })
    })

    expect($desktopOnboarding.get().configured).toBe(false)
  })
})
