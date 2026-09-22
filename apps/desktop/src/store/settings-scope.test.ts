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
const { $connection } = await import('./session')

const {
  $settingsOwner,
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
  $connection.set(null)
})

describe('settings scope store', () => {
  it('captures a new registered owner descriptor when the same id is replaced', () => {
    const original = {
      authMode: 'token',
      baseUrl: 'https://original.example',
      connectionId: 'same-id',
      headers: { 'Cf-Access-Client-Id': 'original-client' },
      mode: 'remote',
      remoteHost: 'operator@original-host',
      token: 'original-token'
    }

    $connection.set(original as never)
    const captured = $settingsOwner.get()

    expect(captured).toMatchObject({ connectionId: 'same-id' })
    expect(captured?.connectionOwner).toMatchObject({
      authMode: original.authMode,
      baseUrl: original.baseUrl,
      headers: original.headers,
      mode: original.mode,
      remoteHost: original.remoteHost,
      token: original.token
    })

    for (const replacement of [
      { ...original, baseUrl: 'https://replacement.example' },
      { ...original, token: 'replacement-token' },
      { ...original, headers: { 'Cf-Access-Client-Id': 'replacement-client' } },
      { ...original, remoteHost: 'operator@replacement-host' }
    ]) {
      $connection.set(original as never)
      const beforeReplacement = $settingsOwner.get()?.connectionOwner
      $connection.set(replacement as never)
      expect($settingsOwner.get()?.connectionOwner).not.toBe(beforeReplacement)
    }
  })

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

  it('exposes a request-shaped scope: the concrete profile, never null', () => {
    // api/client.ts profileScoped() treats null as "target primary/default" —
    // the #90549 bug class. The request form must therefore never be null.
    expect($settingsRequestProfile.get()).toBe('default')

    setSettingsScope('research')
    expect($settingsRequestProfile.get()).toBe('research')

    setSettingsScope('default')
    expect($settingsRequestProfile.get()).toBe('default')
  })

  it('names the profile the page displays even when that IS the active profile', () => {
    // #118432: the "Changes on this page apply to 'x'" note reads
    // $settingsScopeProfile, so the value handed to API helpers must carry the
    // SAME key. An `undefined` request scope makes profileScoped() omit
    // `?profile=` entirely, and the backend resolves an omitted profile to the
    // home it was LAUNCHED under — not the profile being edited. With a pooled
    // desktop backend (`hermes --profile a serve`, editing b) that mismatch
    // made every settings page read a's values and write them back to a, while
    // the note kept naming b.
    $activeGatewayProfile.set('nash')
    expect($settingsScopeProfile.get()).toBe('nash')
    expect($settingsRequestProfile.get()).toBe('nash')

    // Selecting the active profile stores no override — this is the exact case
    // that used to collapse the request scope to undefined.
    setSettingsScope('nash')
    expect($settingsScopeOverride.get()).toBeNull()
    expect($settingsRequestProfile.get()).toBe('nash')
  })

  it('leaves a custom HERMES_HOME on the ambient request path', () => {
    // `custom` names a home outside profiles/: there is no profile directory to
    // resolve, so the ambient path is the only correct answer.
    $activeGatewayProfile.set('custom')

    expect($settingsScopeProfile.get()).toBe('custom')
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
