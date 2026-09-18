import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ProfileInfo } from '@/types/hermes'

// Keep store/profile's side-effecting imports inert — same seam as
// store/profile.test.ts.
vi.mock('@/store/gateway', () => ({
  $gateway: atom<unknown>(null),
  ensureGatewayForAgent: vi.fn(async () => undefined),
  ensureGatewayForProfile: vi.fn(async () => undefined),
  openGatewayForProfile: vi.fn(async () => undefined)
}))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))

const { $activeGatewayProfile, $profiles } = await import('./profile')

const {
  $settingsRequestProfile,
  $settingsScopeEditsNonDefault,
  $settingsScopeOverride,
  $settingsScopeProfile,
  setSettingsScope
} = await import('./settings-scope')

beforeEach(() => {
  $activeGatewayProfile.set('default')
  $settingsScopeOverride.set(null)
  $profiles.set([])
})

describe('settings scope store', () => {
  it('defaults to following the active gateway profile (no override)', () => {
    expect($settingsScopeOverride.get()).toBeNull()
    expect($settingsScopeProfile.get()).toBe('default')

    $activeGatewayProfile.set('coder')
    expect($settingsScopeProfile.get()).toBe('coder')
  })

  it('stores a concrete override when a non-active profile is selected', () => {
    setSettingsScope('research')

    expect($settingsScopeOverride.get()).toBe('research')
    expect($settingsScopeProfile.get()).toBe('research')
  })

  it('selecting the active profile clears the override instead of pinning it', () => {
    setSettingsScope('research')
    setSettingsScope('default')

    // No override → requests keep their unscoped shape and the scope keeps
    // following the app on future profile switches.
    expect($settingsScopeOverride.get()).toBeNull()
    expect($settingsScopeProfile.get()).toBe('default')
  })

  it('normalizes empty/blank names to the default profile key', () => {
    $activeGatewayProfile.set('coder')
    setSettingsScope('')

    expect($settingsScopeOverride.get()).toBe('default')
    expect($settingsScopeProfile.get()).toBe('default')
  })

  it('exposes a request-shaped scope: undefined (never null) without an override', () => {
    // api/client.ts profileScoped() treats null as "target primary/default" —
    // the #90549 bug class. The request form must therefore never be null.
    expect($settingsRequestProfile.get()).toBeUndefined()

    setSettingsScope('research')
    expect($settingsRequestProfile.get()).toBe('research')

    setSettingsScope('default')
    expect($settingsRequestProfile.get()).toBeUndefined()
  })

  it('flags a non-default edit target whether it comes from the active profile or an override', () => {
    const roster = [
      { is_default: true, name: 'default' },
      { is_default: false, name: 'scout' }
    ] as unknown as ProfileInfo[]

    $profiles.set(roster)

    // Following the active DEFAULT profile → editing the default.
    expect($settingsScopeEditsNonDefault.get()).toBe(false)

    // A Bot Mode chat made the bot the active profile; no override is set,
    // yet the settings pages now edit profiles/scout/config.yaml.
    $activeGatewayProfile.set('scout')
    expect($settingsScopeEditsNonDefault.get()).toBe(true)

    // Explicit override back onto the default → editing the default again.
    setSettingsScope('default')
    expect($settingsScopeEditsNonDefault.get()).toBe(false)
  })

  it('treats an unloaded roster as "default = the root profile", so an unknown default fails loud', () => {
    $activeGatewayProfile.set('scout')
    expect($settingsScopeEditsNonDefault.get()).toBe(true)

    $activeGatewayProfile.set('default')
    expect($settingsScopeEditsNonDefault.get()).toBe(false)
  })

  it('drops the override on an app-wide profile switch', () => {
    setSettingsScope('research')
    expect($settingsScopeOverride.get()).toBe('research')

    // The app re-homes to another profile: a surviving override would keep
    // settings edits silently pointed at the previous target.
    $activeGatewayProfile.set('coder')

    expect($settingsScopeOverride.get()).toBeNull()
    expect($settingsScopeProfile.get()).toBe('coder')
  })
})
