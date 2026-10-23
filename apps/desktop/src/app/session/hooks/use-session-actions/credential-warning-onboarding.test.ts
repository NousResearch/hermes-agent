import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $desktopOnboarding, consumePendingCredentialWarning } from '@/store/onboarding'
import { setOnboardingSurfaceActive } from '@/store/onboarding-presence'

import { applyRuntimeInfo } from './utils'

const initialOnboardingState = $desktopOnboarding.get()

describe('applyRuntimeInfo credential warning while onboarding is open', () => {
  beforeEach(() => {
    consumePendingCredentialWarning()
    $desktopOnboarding.set({ ...initialOnboardingState, reason: null, requested: false })
  })

  afterEach(() => {
    setOnboardingSurfaceActive('onboarding', false)
    consumePendingCredentialWarning()
    $desktopOnboarding.set(initialOnboardingState)
  })

  it('drops the warning: onboarding owns the first run', () => {
    setOnboardingSurfaceActive('onboarding', true)
    applyRuntimeInfo({
      credential_warning: "No API key configured for provider 'openrouter'. First message will fail."
    })

    expect(consumePendingCredentialWarning()).toBeNull()
  })
})
