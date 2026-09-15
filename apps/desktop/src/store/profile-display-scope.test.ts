import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// Regression for the /profile status display under All Profiles. `/profile`
// (bare) answers "which profile does the NEXT new chat use" — that is
// ambientNewChatProfile(), not $activeGatewayProfile (the live gateway's
// realized route, which lags a swap and, under All Profiles, still names the
// last-opened profile). The slash handler reads ambientNewChatProfile(); these
// tests pin the value it reads for the reported click sequences.

const activeGatewayConnectionId = vi.fn<() => null | string>(() => null)

vi.mock('@/store/gateway', () => ({
  $gateway: atom<unknown>({ connectionState: 'open', id: 'sock' }),
  activeGatewayConnectionId,
  ensureGatewayForAgent: vi.fn(async () => true),
  ensureGatewayForProfile: vi.fn(async () => undefined),
  openGatewayForAgent: vi.fn(async () => undefined),
  openGatewayForProfile: vi.fn(async () => undefined),
  openSecondaryCount: vi.fn(() => 0)
}))
vi.mock('@/hermes', () => ({ getProfiles: vi.fn(async () => ({ profiles: [] })), hermesApi: {}, setApiRequestProfile: vi.fn(), STARTUP_REQUEST_TIMEOUT_MS: 1000 }))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))
vi.mock('@/store/cron-model-impact-scope', () => ({ invalidateCronModelImpactScopeState: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))
vi.mock('@/store/pool-limits', () => ({ $poolLimits: atom({}) }))
vi.mock('@/store/profile-remote-override', () => ({ notifyRemoteOverrideAuthFailure: vi.fn(() => false) }))
vi.mock('@/store/session', () => ({ clearComposerSelectionOwner: vi.fn(), setComposerSelectionOwner: vi.fn(), setConnection: vi.fn() }))

const { $activeGatewayProfile, $newChatProfile, ambientNewChatProfile, setShowAllProfiles } = await import('./profile')

// What the /profile slash handler reads for its status line.
const profileDisplay = () => ambientNewChatProfile()

beforeEach(() => {
  activeGatewayConnectionId.mockReturnValue(null)
  $activeGatewayProfile.set('default')
  $newChatProfile.set(null)
  setShowAllProfiles(false)
})

describe('/profile display value under All Profiles', () => {
  it('shows the concrete profile when a profile is active', () => {
    $activeGatewayProfile.set('felix')
    expect(profileDisplay()).toBe('felix')
  })

  it('shows default under All Profiles even if felix was the last-opened profile', () => {
    // felix selected, gateway homed on felix, then user switches to All Profiles
    $activeGatewayProfile.set('felix')
    setShowAllProfiles(true)
    expect(profileDisplay()).toBe('default')
  })

  it('shows default under All Profiles after tommy (no jump to a stale profile)', () => {
    $activeGatewayProfile.set('tommy')
    setShowAllProfiles(true)
    expect(profileDisplay()).toBe('default')
  })
})
