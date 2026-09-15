import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// A plain new chat (the "New session" row, no per-profile "+") carries no
// explicit profile intent. While browsing "All Profiles" there is NO single
// active context to inherit: the sidebar is served off every profile's
// databases at once, and $activeGatewayProfile still names whichever profile
// was opened most recently. The old fallback resolved the owner from that, so
// a fresh chat started under All Profiles silently landed in the last-opened
// profile (e.g. "felix") instead of the primary "default". It must fall to the
// primary/default door instead — the same convention cron already uses
// (ALL_PROFILES -> writable profile "default").

const activeGatewayConnectionId = vi.fn<() => null | string>(() => null)

vi.mock('@/store/gateway', () => ({
  $gateway: atom<unknown>({ id: 'live-socket' }),
  activeGatewayConnectionId,
  ensureGatewayForAgent: vi.fn(async () => true),
  ensureGatewayForProfile: vi.fn(async () => undefined),
  openGatewayForAgent: vi.fn(async () => undefined),
  openGatewayForProfile: vi.fn(async () => undefined),
  openSecondaryCount: vi.fn(() => 0)
}))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  hermesApi: {},
  setApiRequestProfile: vi.fn(),
  STARTUP_REQUEST_TIMEOUT_MS: 1000
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))
vi.mock('@/store/cron-model-impact-scope', () => ({ invalidateCronModelImpactScopeState: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))
vi.mock('@/store/pool-limits', () => ({ $poolLimits: atom({}) }))
vi.mock('@/store/profile-remote-override', () => ({ notifyRemoteOverrideAuthFailure: vi.fn(() => false) }))
vi.mock('@/store/session', () => ({
  clearComposerSelectionOwner: vi.fn(),
  setComposerSelectionOwner: vi.fn(),
  setConnection: vi.fn()
}))

const {
  $activeGatewayProfile,
  $newChatConnectionId,
  $newChatProfile,
  $newChatRoute,
  ambientNewChatProfile,
  resolveNewChatOwnerRoute,
  setShowAllProfiles
} = await import('./profile')

beforeEach(() => {
  activeGatewayConnectionId.mockReset()
  activeGatewayConnectionId.mockReturnValue(null)
  $activeGatewayProfile.set('felix')
  $newChatProfile.set(null)
  $newChatRoute.set(null)
  $newChatConnectionId.set(null)
  setShowAllProfiles(false)
})

describe('plain new chat under All Profiles', () => {
  it('resolves the ambient profile to default, not the last-active profile', () => {
    setShowAllProfiles(true)

    expect(ambientNewChatProfile()).toBe('default')
  })

  it('follows the live gateway profile when NOT browsing all profiles', () => {
    setShowAllProfiles(false)

    expect(ambientNewChatProfile()).toBe('felix')
  })

  it('takes the legacy default door for an intent-less chat under all profiles', () => {
    activeGatewayConnectionId.mockReturnValue('felix-socket')
    setShowAllProfiles(true)

    expect(resolveNewChatOwnerRoute()).toBeNull()
  })

  it('still honors an explicit per-profile "+" intent under all profiles', () => {
    activeGatewayConnectionId.mockReturnValue('mini')
    setShowAllProfiles(true)
    $newChatProfile.set('researcher')
    $newChatConnectionId.set('mini')

    expect(resolveNewChatOwnerRoute()).toEqual({ connectionId: 'mini', profile: 'researcher' })
  })
})
