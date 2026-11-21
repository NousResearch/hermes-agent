import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { requestGatewayForProfile } from './gateway'
import { setPetInfo } from './pet'
import {
  $petGallery,
  $petGalleryError,
  $petGalleryStatus,
  adoptPet,
  applyAdoptedPet,
  exportPet,
  type GatewayRequest,
  loadPetGallery,
  loadPetThumb,
  type PetGallery,
  removePet,
  renamePet,
  resetPetGallery,
  setPetEnabled,
  setPetScale
} from './pet-gallery'
import { $activeGatewayProfile } from './profile'
import { $activeSessionId, $selectedStoredSessionId, $sessions } from './session'

// A non-visible owner's call leaves on that profile's own socket, not the
// ambient request; record it so a test can see exactly where a write went.
vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  requestGatewayForProfile: vi.fn(async () => ({ ok: true }))
}))

const routed = vi.mocked(requestGatewayForProfile)

const gallery = (name: string): PetGallery => ({
  enabled: true,
  active: 'same-slug',
  pets: [{ slug: 'same-slug', displayName: name, installed: true }]
})

beforeEach(() => {
  vi.useFakeTimers()
  $sessions.set([])
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $activeGatewayProfile.set('owner-a')
  resetPetGallery()
  routed.mockClear()
  setPetInfo({ enabled: true, slug: 'same-slug', spritesheetBase64: 'A', scale: 1 })
})

afterEach(() => {
  resetPetGallery()
  vi.clearAllTimers()
  vi.useRealTimers()
  $activeGatewayProfile.set('default')
  setPetInfo({ enabled: false })
})

describe('owner switch inside the resize debounce window', () => {
  it("persists A's pending resize to A and sends nothing to B", async () => {
    const ambient = vi.fn(async () => ({ ok: true, scale: 2 })) as unknown as GatewayRequest
    setPetScale(ambient, 2)
    await vi.advanceTimersByTimeAsync(50)
    $activeGatewayProfile.set('owner-b')
    await vi.advanceTimersByTimeAsync(500)

    expect(routed).toHaveBeenCalledTimes(1)
    expect(routed).toHaveBeenCalledWith('owner-a', 'pet.scale', { scale: 2, profile: 'owner-a' }, undefined, undefined)
    // B is now the ambient owner: no scale write may reach it.
    expect(ambient).not.toHaveBeenCalled()
  })

  it('flushes only the last value of a drag, once', async () => {
    const ambient = vi.fn(async () => ({ ok: true })) as unknown as GatewayRequest
    setPetScale(ambient, 1.5)
    setPetScale(ambient, 2)
    $activeGatewayProfile.set('owner-b')
    await vi.advanceTimersByTimeAsync(500)

    expect(routed.mock.calls.map(call => call.slice(0, 3))).toEqual([
      ['owner-a', 'pet.scale', { scale: 2, profile: 'owner-a' }]
    ])
  })
})

describe('gallery load in flight across an owner switch', () => {
  it.each([
    ['success', (settle: Settle) => settle.resolve(gallery('A late'))],
    ['failure', (settle: Settle) => settle.reject(new Error('A late failure'))]
  ])("publishes neither A's late %s into B's gallery", async (_label, finish) => {
    const settle = deferred()
    const ambient = vi.fn(() => settle.promise) as unknown as GatewayRequest
    $petGallery.set(gallery('A cached'))
    const operation = loadPetGallery(ambient, { force: true })
    expect(ambient).toHaveBeenCalled()

    // The switch alone must cancel A's load; no manual reset in between.
    $activeGatewayProfile.set('owner-b')
    const b = gallery('B own')
    $petGallery.set(b)
    $petGalleryStatus.set('ready')
    finish(settle)
    await operation

    expect($petGallery.get()).toEqual(b)
    expect($petGalleryStatus.get()).toBe('ready')
    expect($petGalleryError.get()).toBeNull()
  })
})

// Every exported gallery action must route through the visible owner. A call
// site that bypasses the owner wrapper sends no `profile` and fails here.
describe('no unwrapped pet RPC in the gallery store', () => {
  const actions: Array<[string, (request: GatewayRequest) => Promise<unknown> | void]> = [
    ['loadPetGallery', request => loadPetGallery(request, { force: true })],
    ['loadPetThumb', request => loadPetThumb(request, 'thumb-slug')],
    ['adoptPet', request => adoptPet(request, 'same-slug', 'failed')],
    ['applyAdoptedPet', request => applyAdoptedPet(request, 'same-slug', 'Same')],
    ['setPetEnabled(true)', request => setPetEnabled(request, true, { noneAvailable: 'none', fallback: 'failed' })],
    ['setPetEnabled(false)', request => setPetEnabled(request, false, { noneAvailable: 'none', fallback: 'failed' })],
    ['removePet', request => removePet(request, 'same-slug', 'failed')],
    ['exportPet', request => exportPet(request, 'same-slug', 'failed')],
    ['renamePet', request => renamePet(request, 'same-slug', 'Renamed', 'failed')],
    ['setPetScale', request => setPetScale(request, 2)]
  ]

  it.each(actions)('%s stamps the visible owner on every call', async (_label, action) => {
    const calls: Array<[string, Record<string, unknown> | undefined]> = []

    const ambient = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push([method, params])

      return { ok: true, enabled: true, active: 'same-slug', pets: [], slug: 'same-slug', displayName: 'Same' }
    }) as unknown as GatewayRequest

    $petGallery.set(gallery('A'))
    await action(ambient)
    await vi.advanceTimersByTimeAsync(500)

    expect(calls.length).toBeGreaterThan(0)

    for (const [method, params] of calls) {
      expect(params?.profile, `${method} carries the visible owner`).toBe('owner-a')
    }
  })
})

interface Settle {
  resolve: (value: PetGallery) => void
  reject: (reason: Error) => void
}

function deferred(): Settle & { promise: Promise<PetGallery> } {
  let resolve!: Settle['resolve']
  let reject!: Settle['reject']

  const promise = new Promise<PetGallery>((done, fail) => {
    resolve = done
    reject = fail
  })

  return { promise, resolve, reject }
}
