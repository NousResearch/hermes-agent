import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// Picking a profile must stay on the source the user is LOOKING at. $profiles
// is the active gateway's list, so a pick made while a registry source is live
// names one of THAT source's profiles. Routing it through the profile-only
// path resolved the descriptor with a bare name, which the main process
// answers against the primary — the gateway snapped back home and the pick
// looked like it never took.

const ensureGatewayForProfile = vi.fn(async (_profile: string) => undefined)
const ensureGatewayForAgent = vi.fn(async (_connectionId: null | string, _profile: string) => true)
const openGatewayForProfile = vi.fn(async (_profile: string) => undefined)
const activeGatewayConnectionId = vi.fn<() => null | string>(() => null)
const $gateway = atom<unknown>({ id: 'live-socket' })
const resetStarmapGraph = vi.fn()

vi.mock('@/store/gateway', () => ({
  $gateway,
  activeGatewayConnectionId,
  ensureGatewayForAgent,
  ensureGatewayForProfile,
  openGatewayForProfile
}))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph }))

const { $activeGatewayProfile, newSessionInProfile, selectProfile } = await import('./profile')

beforeEach(() => {
  ensureGatewayForProfile.mockClear()
  ensureGatewayForAgent.mockClear()
  activeGatewayConnectionId.mockReset()
  activeGatewayConnectionId.mockReturnValue(null)
  $gateway.set({ id: 'live-socket' })
  $activeGatewayProfile.set('default')
  // resolveConnectionForAgent is best-effort; without a bridge it resolves
  // null and the previous descriptor stays, which is fine here.
  ;(globalThis as { window?: unknown }).window = {}
})

describe('selectProfile', () => {
  it('activates the pick on the live registry source, not the primary', async () => {
    activeGatewayConnectionId.mockReturnValue('mini')

    selectProfile('researcher')

    await vi.waitFor(() => expect(ensureGatewayForAgent).toHaveBeenCalledWith('mini', 'researcher'))
    expect(ensureGatewayForProfile).not.toHaveBeenCalled()
  })

  it('keeps the legacy profile-only path when the primary is live', async () => {
    activeGatewayConnectionId.mockReturnValue(null)

    selectProfile('ops')

    await vi.waitFor(() => expect(ensureGatewayForProfile).toHaveBeenCalledWith('ops'))
    expect(ensureGatewayForAgent).not.toHaveBeenCalled()
  })

  it('keeps the legacy profile-only path when the explicit local source is live', async () => {
    activeGatewayConnectionId.mockReturnValue('local')

    selectProfile('override-profile')

    await vi.waitFor(() => expect(ensureGatewayForProfile).toHaveBeenCalledWith('override-profile'))
    expect(ensureGatewayForAgent).not.toHaveBeenCalled()
  })
})

describe('newSessionInProfile', () => {
  it('opens the new chat on the live registry source', async () => {
    activeGatewayConnectionId.mockReturnValue('mini')

    newSessionInProfile('designer')

    await vi.waitFor(() => expect(ensureGatewayForAgent).toHaveBeenCalledWith('mini', 'designer'))
    expect(ensureGatewayForProfile).not.toHaveBeenCalled()
  })

  it('keeps the legacy profile-only path for a new chat on the explicit local source', async () => {
    activeGatewayConnectionId.mockReturnValue('local')

    newSessionInProfile('override-profile')

    await vi.waitFor(() => expect(ensureGatewayForProfile).toHaveBeenCalledWith('override-profile'))
    expect(ensureGatewayForAgent).not.toHaveBeenCalled()
  })
})
