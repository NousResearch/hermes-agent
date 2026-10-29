import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { persistBoolean, provisionalPreferenceKeys } from '@/lib/storage'

// Keep profile.ts's side-effecting imports inert (same pattern as
// profile.test.ts): the gateway socket layer and REST client must not run in a
// unit test.
const $gateway = atom<unknown>({ id: 'live-socket', connectionState: 'open' })

vi.mock('@/store/gateway', () => ({
  $gateway,
  activeGateway: () => null,
  activeGatewayConnectionId: () => null,
  activeGatewayProfileKey: () => 'default',
  ensureGatewayForProfile: vi.fn(async () => undefined),
  openGatewayForProfile: vi.fn(async () => undefined),
  openSecondaryCount: () => 0
}))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ reloadOpenStarmapGraph: vi.fn(), resetStarmapGraph: vi.fn() }))

const { $showAllProfiles, applyUnifiedInboxDefault, setShowAllProfiles } = await import('./profile')

const KEY = 'hermes.desktop.showAllProfiles'

/** A client that has never expressed a scope: the store's own default (false),
 * stored provisionally — exactly what persistStore leaves after a plain load. */
function freshClient() {
  $showAllProfiles.set(false)
  window.localStorage.clear()
  persistBoolean(KEY, false, { isDefault: true, provisional: true })
}

const isProvisional = () => provisionalPreferenceKeys().has(KEY)

describe('applyUnifiedInboxDefault — unified view by default for a multi-connection registry', () => {
  beforeEach(freshClient)

  it('turns the unified view on for a registry with more than one connection', () => {
    applyUnifiedInboxDefault(2)

    expect($showAllProfiles.get()).toBe(true)
  })

  it('leaves a single-connection registry on the legacy scope', () => {
    applyUnifiedInboxDefault(1)
    applyUnifiedInboxDefault(0)

    expect($showAllProfiles.get()).toBe(false)
    expect(window.localStorage.getItem(KEY)).toBe('false')
  })

  it('is idempotent: re-publishing the same registry changes nothing', () => {
    applyUnifiedInboxDefault(3)
    const stored = window.localStorage.getItem(KEY)

    applyUnifiedInboxDefault(3)
    applyUnifiedInboxDefault(2)

    expect($showAllProfiles.get()).toBe(true)
    expect(window.localStorage.getItem(KEY)).toBe(stored)
  })

  it('writes the default provisionally, so the machine-shared preferences never take it as a choice', () => {
    applyUnifiedInboxDefault(2)

    expect(window.localStorage.getItem(KEY)).toBe('true')
    expect(isProvisional()).toBe(true)
  })

  it('follows the registry back down while the value is still only a default', () => {
    applyUnifiedInboxDefault(2)
    applyUnifiedInboxDefault(1)

    expect($showAllProfiles.get()).toBe(false)
    expect(isProvisional()).toBe(true)
  })
})

describe('applyUnifiedInboxDefault — the user always wins', () => {
  beforeEach(freshClient)

  it('an explicit opt-out survives every later registry publication', () => {
    applyUnifiedInboxDefault(2)
    expect($showAllProfiles.get()).toBe(true)

    // The user turns the unified view off: from here the value is theirs.
    setShowAllProfiles(false)
    expect(isProvisional()).toBe(false)

    applyUnifiedInboxDefault(2)
    applyUnifiedInboxDefault(5)

    expect($showAllProfiles.get()).toBe(false)
    expect(window.localStorage.getItem(KEY)).toBe('false')
  })

  it('an explicit opt-in is never turned off, even when the registry drops to one connection', () => {
    setShowAllProfiles(true)
    expect(isProvisional()).toBe(false)

    applyUnifiedInboxDefault(1)

    expect($showAllProfiles.get()).toBe(true)
  })

  it('an opt-in made before any registry publication is kept, not re-derived', () => {
    setShowAllProfiles(true)

    applyUnifiedInboxDefault(2)

    expect($showAllProfiles.get()).toBe(true)
    expect(isProvisional()).toBe(false)
  })
})
