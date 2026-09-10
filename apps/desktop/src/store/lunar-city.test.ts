import { beforeEach, describe, expect, it } from 'vitest'

import {
  $worldEnabled,
  $worldOnboardingDismissed,
  setWorldEnabled,
  setWorldOnboardingDismissed,
  WORLD_ENABLED_STORAGE_KEY,
  WORLD_ONBOARDING_DISMISSED_STORAGE_KEY
} from './lunar-city'

describe('Lunar City desktop state', () => {
  beforeEach(() => {
    localStorage.clear()
    $worldEnabled.set(true)
    $worldOnboardingDismissed.set(false)
  })

  it('persists the Enable World preference', () => {
    setWorldEnabled(false)

    expect($worldEnabled.get()).toBe(false)
    expect(localStorage.getItem(WORLD_ENABLED_STORAGE_KEY)).toBe('false')

    setWorldEnabled(true)

    expect(localStorage.getItem(WORLD_ENABLED_STORAGE_KEY)).toBe('true')
  })

  it('persists dismissal of the first-open onboarding', () => {
    setWorldOnboardingDismissed()

    expect($worldOnboardingDismissed.get()).toBe(true)
    expect(localStorage.getItem(WORLD_ONBOARDING_DISMISSED_STORAGE_KEY)).toBe('true')
  })
})
