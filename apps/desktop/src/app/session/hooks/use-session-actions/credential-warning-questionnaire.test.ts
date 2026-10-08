import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $desktopOnboarding, consumePendingCredentialWarning } from '@/store/onboarding'
import { setOnboardingSurfaceActive } from '@/store/onboarding-presence'

import { applyRuntimeInfo } from './utils'

const initialOnboardingState = $desktopOnboarding.get()

describe('applyRuntimeInfo credential warning while the questionnaire is open', () => {
  beforeEach(() => {
    consumePendingCredentialWarning()
    $desktopOnboarding.set({ ...initialOnboardingState, reason: null, requested: false })
  })

  afterEach(() => {
    setOnboardingSurfaceActive('questionnaire', false)
    consumePendingCredentialWarning()
    $desktopOnboarding.set(initialOnboardingState)
  })

  it('drops the warning: the questionnaire owns the first run', () => {
    setOnboardingSurfaceActive('questionnaire', true)
    applyRuntimeInfo({
      credential_warning: "No API key configured for provider 'openrouter'. First message will fail."
    })

    expect(consumePendingCredentialWarning()).toBeNull()
  })
})
