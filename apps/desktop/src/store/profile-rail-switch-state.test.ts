import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile, $freshSessionRequest, newSessionInProfile, selectProfile } from './profile'
import { $profileSwitchFailure, $profileSwitchPending } from './profile-switch-state'

// A rejected rail switch must end in an explicit, retryable failure the chat
// can show — not only a transient toast over an empty draft (#81094) — and
// Retry must re-run the activation, not start another fresh draft.

const { ensureGatewayForProfile, notifyError } = vi.hoisted(() => ({
  ensureGatewayForProfile: vi.fn(async (_profile: string): Promise<undefined> => undefined),
  notifyError: vi.fn()
}))

vi.mock('@/store/gateway', () => ({
  $gateway: atom<unknown>({ connectionState: 'open' }),
  activeGatewayConnectionId: () => null,
  activeGatewayProfileKey: () => ensureGatewayForProfile.mock.lastCall?.[0] ?? 'default',
  ensureGatewayForAgent: vi.fn(),
  ensureGatewayForProfile,
  openGatewayForProfile: vi.fn()
}))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))
vi.mock('@/store/starmap', () => ({ reloadOpenStarmapGraph: vi.fn(), resetStarmapGraph: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError }))

beforeEach(() => {
  ensureGatewayForProfile.mockReset()
  notifyError.mockClear()
  $activeGatewayProfile.set('default')
  $profileSwitchFailure.set(null)
  $profileSwitchPending.set(null)
  ;(globalThis as { window?: unknown }).window = {}
})

describe('rail switch failure', () => {
  it('ends a rejected selectProfile in a retryable failure beside the toast', async () => {
    ensureGatewayForProfile.mockRejectedValueOnce(new Error('dial timed out'))

    selectProfile('ops')
    expect($profileSwitchPending.get()).toBe('ops')

    await vi.waitFor(() => expect($profileSwitchFailure.get()?.profile).toBe('ops'))
    expect($profileSwitchPending.get()).toBeNull()
    expect(notifyError).toHaveBeenCalledWith(expect.any(Error), 'Failed to switch to profile "ops"')

    // Retry re-dials on the original pick; success clears the failure, and no
    // second fresh draft is requested.
    const drafts = $freshSessionRequest.get()
    ensureGatewayForProfile.mockResolvedValueOnce(undefined)
    $profileSwitchFailure.get()!.retry()

    await vi.waitFor(() => expect(ensureGatewayForProfile).toHaveBeenCalledTimes(2))
    await vi.waitFor(() => expect($profileSwitchPending.get()).toBeNull())
    expect($profileSwitchFailure.get()).toBeNull()
    expect($freshSessionRequest.get()).toBe(drafts)
  })

  it('does the same for newSessionInProfile', async () => {
    ensureGatewayForProfile.mockRejectedValueOnce(new Error('refused'))

    newSessionInProfile('designer')

    await vi.waitFor(() => expect($profileSwitchFailure.get()?.profile).toBe('designer'))
  })

  it('lets a superseded attempt neither clear the newer spinner nor resurrect a failure', async () => {
    const first = Promise.withResolvers<undefined>()
    ensureGatewayForProfile.mockReturnValueOnce(first.promise)
    ensureGatewayForProfile.mockReturnValueOnce(Promise.withResolvers<undefined>().promise)

    selectProfile('ops')
    await vi.waitFor(() => expect(ensureGatewayForProfile).toHaveBeenCalledTimes(1))
    selectProfile('designer')
    expect($profileSwitchPending.get()).toBe('designer')

    first.reject(new Error('late'))
    await vi.waitFor(() => expect(notifyError).toHaveBeenCalled())

    expect($profileSwitchPending.get()).toBe('designer')
    expect($profileSwitchFailure.get()).toBeNull()
  })
})
