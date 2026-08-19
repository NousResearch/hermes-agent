import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { GatewayRequest } from './pet-gallery'

// Keep the heavy, side-effecting import chains inert for a unit test — only
// pet.ts / profile.ts / pet-gallery.ts / pet-generate.ts run for real, so
// petProfile()'s actual resolution (the thing this test exists to check) is
// genuinely exercised rather than mocked away.
const $gateway = atom<unknown>({ id: 'live-socket', on: vi.fn(() => () => undefined) })
vi.mock('@/store/gateway', () => ({
  $gateway,
  ensureGatewayForProfile: vi.fn(async () => undefined),
  openGatewayForProfile: vi.fn(async () => undefined)
}))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn(),
  STARTUP_REQUEST_TIMEOUT_MS: 5000
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))
vi.mock('@/store/native-notifications', () => ({ dispatchNativeNotification: vi.fn() }))

const { checkPetGenAvailable, generateDrafts, discardHatched } = await import('./pet-generate')
const { $activeGatewayProfile } = await import('./profile')

/** Every pet.* RPC must resolve against the desktop's active (possibly
 *  non-launch) profile — see pet.ts's petProfile() docstring. Regression
 *  coverage for the pet-generate.ts call sites that used to call the bare
 *  `request()` passed into them, silently dropping `profile` and always
 *  hitting the launch profile's HERMES_HOME regardless of which profile the
 *  desktop UI was actually showing. */
describe('pet-generate.ts RPC calls carry the active profile', () => {
  beforeEach(() => {
    $activeGatewayProfile.set('beta')
  })

  it('pet.generate.status carries profile', async () => {
    const request = vi.fn(async () => ({ available: false, providers: [] })) as unknown as GatewayRequest

    await checkPetGenAvailable(request)

    expect(request).toHaveBeenCalledWith(
      'pet.generate.status',
      expect.objectContaining({ profile: 'beta' }),
      undefined,
      undefined
    )
  })

  it('pet.generate carries profile alongside its long timeout + abort signal', async () => {
    const request = vi.fn(async () => ({
      ok: true,
      token: 'tok1',
      drafts: [{ index: 0, dataUri: 'data:x' }]
    })) as unknown as GatewayRequest

    await generateDrafts(request, { prompt: 'a fox' })

    expect(request).toHaveBeenCalledWith(
      'pet.generate',
      expect.objectContaining({ profile: 'beta', prompt: 'a fox' }),
      expect.any(Number),
      expect.any(AbortSignal)
    )
  })

  it('pet.remove (discard) carries profile', async () => {
    const request = vi.fn(async () => ({ ok: true })) as unknown as GatewayRequest
    const { $petGenPreview } = await import('./pet-generate')
    $petGenPreview.set({ slug: 'stray-pet', displayName: 'Stray', enabled: false })

    await discardHatched(request)

    expect(request).toHaveBeenCalledWith(
      'pet.remove',
      expect.objectContaining({ profile: 'beta', slug: 'stray-pet' }),
      undefined,
      undefined
    )
  })
})
