import { afterEach, expect, it, onTestFinished, vi } from 'vitest'

import { registry } from '@/contrib/registry'

import { registerLayoutPresets } from './layout-presets'

afterEach(() => vi.unstubAllGlobals())

it('keeps the main layout set with onboarding disabled', () => {
  vi.stubGlobal('hermesDesktop', { guestOnboardingEnabled: false })
  // The layout invariant does not need the controller's UI imports or watchers.
  onTestFinished(registerLayoutPresets())
  const layouts = registry.getArea('layouts').map(preset => preset.id)
  expect(new Set(layouts)).toEqual(new Set(['default', 'focus', 'terminal-deck', 'quad']))
  expect(layouts).not.toContain('basic')
})
