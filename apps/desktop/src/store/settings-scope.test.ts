import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

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

const { $activeGatewayProfile } = await import('./profile')
const { $settingsScopeOverride, $settingsScopeProfile, setSettingsScope } = await import('./settings-scope')

beforeEach(() => {
  $activeGatewayProfile.set('default')
  $settingsScopeOverride.set(null)
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
