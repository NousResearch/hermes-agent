import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// profile.ts's side-effecting imports must stay inert: the gateway socket layer and
// the REST query client must not run for real in a unit test.
vi.mock('@/store/gateway', () => ({
  $gateway: atom({ id: 'live-socket', connectionState: 'open' }),
  activeGateway: () => null,
  activeGatewayConnectionId: () => null,
  activeGatewayProfileKey: () => $activeGatewayProfile.get(),
  ensureGatewayForAgent: vi.fn(),
  ensureGatewayForProfile: vi.fn(),
  openGatewayForAgent: vi.fn(),
  openGatewayForProfile: vi.fn(),
  openSecondaryCount: vi.fn(() => 0)
}))
vi.mock('@/store/pool-limits', async () => {
  const { atom } = await import('nanostores')

  return { $poolLimits: atom({ idleMs: 600_000, maxBackends: 3 }) }
})
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))

const { $activeGatewayProfile, $profileColors, $profileOrder, renameProfileInRailPrefs } = await import('./profile')

// A rename moves the profile's directory, so the rail preferences keyed by the old
// name have to move with it. The stored order is exactly what ⌘N resolves against
// (switchProfileToSlot → sortByProfileOrder), so a name that stops matching drops the
// profile into the alphabetical tail and silently reassigns every slot below it
// (#130397). The long-press colour is keyed the same way and dies with the old name.
describe('renameProfileInRailPrefs', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $activeGatewayProfile.set('default')
    $profileOrder.set([])
    $profileColors.set({})
  })

  it('keeps the renamed profile in its slot instead of dropping it to the tail', () => {
    $profileOrder.set(['scout', 'editor', 'qa'])

    renameProfileInRailPrefs('editor', 'house')

    expect($profileOrder.get()).toEqual(['scout', 'house', 'qa'])
  })

  it('carries the long-press colour override across the rename', () => {
    $profileColors.set({ editor: 'coral' })

    renameProfileInRailPrefs('editor', 'house')

    expect($profileColors.get()).toEqual({ house: 'coral' })
  })

  it('leaves a profile that was never in the stored order alone', () => {
    $profileOrder.set(['scout'])
    $profileColors.set({})

    renameProfileInRailPrefs('newcomer', 'house')

    expect($profileOrder.get()).toEqual(['scout'])
  })

  it('drops the old slot rather than duplicating it when the new name already holds one', () => {
    $profileOrder.set(['scout', 'house', 'editor'])
    $profileColors.set({ editor: 'coral', house: 'teal' })

    renameProfileInRailPrefs('editor', 'house')

    expect($profileOrder.get()).toEqual(['scout', 'house'])
    // The surviving name is the surviving identity, so its own colour wins.
    expect($profileColors.get()).toEqual({ house: 'teal' })
  })

  it('is a no-op when the rename leaves the name unchanged', () => {
    $profileOrder.set(['scout', 'editor'])
    $profileColors.set({ editor: 'coral' })

    renameProfileInRailPrefs('editor', 'editor')

    expect($profileOrder.get()).toEqual(['scout', 'editor'])
    expect($profileColors.get()).toEqual({ editor: 'coral' })
  })
})
