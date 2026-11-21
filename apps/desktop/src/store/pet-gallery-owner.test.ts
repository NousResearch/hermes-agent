import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $petInfo, petProfile, setPetInfo } from './pet'
import {
  $petBusy,
  $petGallery,
  $petGalleryError,
  $petGalleryStatus,
  adoptPet,
  exportPet,
  type GatewayRequest,
  loadPetGallery,
  removePet,
  renamePet,
  resetPetGallery,
  setPetEnabled,
  setPetScale
} from './pet-gallery'
import { $activeGatewayProfile } from './profile'
import { $activeSessionId, $selectedStoredSessionId, $sessions } from './session'
import { recordSessionEventScope } from './session-states'

const gallery = (name: string) => ({
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
  $petGallery.set(gallery('A original'))
  setPetInfo({ enabled: true, slug: 'same-slug', spritesheetBase64: 'A', scale: 1 })
})

afterEach(() => {
  resetPetGallery()
  vi.clearAllTimers()
  vi.useRealTimers()
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $sessions.set([])
  $activeGatewayProfile.set('default')
  setPetInfo({ enabled: false })
})

describe('pet owner isolation through asynchronous callers', () => {
  it.each([false, true])('cancels a queued resize on an owner switch (switch back: %s)', async switchBack => {
    const request = vi.fn(async () => ({ ok: true, scale: 2 })) as unknown as GatewayRequest
    setPetScale(request, 2)
    $activeGatewayProfile.set('owner-b')

    if (switchBack) {
      $activeGatewayProfile.set('owner-a')
    }

    await vi.advanceTimersByTimeAsync(250)
    expect(request).not.toHaveBeenCalled()
  })

  it('resolves a runtime-only chat from its event owner instead of the launch profile', () => {
    recordSessionEventScope({ session_id: 'runtime-pet-b', connectionId: 'remote-family', profile: 'owner-b' })
    $activeSessionId.set('runtime-pet-b')
    expect(petProfile()).toBe('owner-b')
  })

  it('reacts when an event proves the owner after the runtime was already focused', () => {
    $activeSessionId.set('runtime-pet-late-owner')
    expect(petProfile()).toBe('owner-a')
    recordSessionEventScope({ session_id: 'runtime-pet-late-owner', connectionId: 'remote-family', profile: 'owner-b' })
    expect(petProfile()).toBe('owner-b')
  })

  it('does not publish an old load failure into an empty new gallery', async () => {
    let reject!: (error: Error) => void

    const response = new Promise<never>((_done, fail) => {
      reject = fail
    })

    const request = vi.fn(() => response) as unknown as GatewayRequest
    resetPetGallery()
    const operation = loadPetGallery(request)
    $activeGatewayProfile.set('owner-b')
    resetPetGallery()
    $petGalleryStatus.set('loading')
    reject(new Error('old gallery failure'))
    await operation
    expect($petGallery.get()).toBeNull()
    expect($petGalleryError.get()).toBeNull()
    expect($petGalleryStatus.get()).toBe('loading')
  })

  it('does not reconcile an in-flight resize into the new owner sprite', async () => {
    let resolve!: (value: { ok: boolean; scale: number }) => void

    const response = new Promise<{ ok: boolean; scale: number }>(done => {
      resolve = done
    })

    const request = vi.fn(() => response) as unknown as GatewayRequest
    setPetScale(request, 2)
    await vi.advanceTimersByTimeAsync(200)
    expect(request).toHaveBeenCalledWith('pet.scale', { scale: 2, profile: 'owner-a' })
    $activeGatewayProfile.set('owner-b')
    setPetInfo({ enabled: true, slug: 'B', spritesheetBase64: 'B', scale: 0.3 })
    resolve({ ok: true, scale: 2 })
    await vi.advanceTimersByTimeAsync(0)
    expect($petInfo.get().scale).toBe(0.3)
  })

  it('does not patch a new gallery when the owner switches between RPC completion and its caller', async () => {
    const request = vi.fn(async () => ({ ok: true })) as unknown as GatewayRequest
    const operation = adoptPet(request, 'a-only', 'failed')
    const current = gallery('B original')
    queueMicrotask(() => {
      $activeGatewayProfile.set('owner-b')
      $petGallery.set(current)
    })
    await operation
    expect($petGallery.get()).toEqual(current)
  })

  // Gallery load is covered mid-flight (no manual reset) in pet-gallery-owner-switch.test.ts.
  const actions: Array<[string, (request: GatewayRequest) => Promise<unknown>]> = [
    ['adopt', request => adoptPet(request, 'same-slug', 'failed')],
    ['enable', request => setPetEnabled(request, true, { noneAvailable: 'none', fallback: 'failed' })],
    ['disable', request => setPetEnabled(request, false, { noneAvailable: 'none', fallback: 'failed' })],
    ['remove', request => removePet(request, 'same-slug', 'failed')],
    ['export', request => exportPet(request, 'same-slug', 'failed')],
    ['rename', request => renamePet(request, 'same-slug', 'A changed', 'failed')]
  ]

  it.each(actions)('silently drops %s cancellation without changing the new owner', async (_label, action) => {
    let reject!: (reason: Error) => void

    const response = new Promise<never>((_resolve, fail) => {
      reject = fail
    })

    const request = vi.fn(() => response) as unknown as GatewayRequest
    const operation = action(request)
    $activeGatewayProfile.set('owner-b')
    resetPetGallery()
    const b = gallery('B original')
    $petGallery.set(b)
    $petGalleryStatus.set('ready')
    $petBusy.set('B operation')
    reject(new Error('old transport failed'))
    await operation
    expect($petGallery.get()).toEqual(b)
    expect($petGalleryError.get()).toBeNull()
    expect($petGalleryStatus.get()).toBe('ready')
    expect($petBusy.get()).toBe('B operation')
  })

  it('rejects a successful old response after an A to B to A switch', async () => {
    let resolve!: (value: { ok: boolean }) => void

    const response = new Promise<{ ok: boolean }>(done => {
      resolve = done
    })

    const request = vi.fn(() => response) as unknown as GatewayRequest
    const operation = adoptPet(request, 'same-slug', 'failed')
    $activeGatewayProfile.set('owner-b')
    resetPetGallery()
    $activeGatewayProfile.set('owner-a')
    const current = gallery('A new view')
    $petGallery.set(current)
    resolve({ ok: true })
    await operation
    expect($petGallery.get()).toEqual(current)
    expect($petGalleryError.get()).toBeNull()
    expect($petInfo.get().enabled).toBe(false)
  })
})
