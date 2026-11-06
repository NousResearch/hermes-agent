import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $changeEventsAvailable, $petChange, notifyPetChanged } from '@/store/live-sync'
import { type PetInfo, setPetInfo } from '@/store/pet'
import { $petGallery, resetPetGallery } from '@/store/pet-gallery'
import { $petOverlayActive } from '@/store/pet-overlay'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $gatewayState, $selectedStoredSessionId, $sessions } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { FloatingPet } from './floating-pet'

const { rpc } = vi.hoisted(() => ({ rpc: vi.fn() }))
vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway: rpc })
}))
vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  activeGatewayConnectionId: () => 'local',
  requestGatewayForAgent: (_connection: string, _profile: string, method: string, params: Record<string, unknown>) =>
    rpc(method, params)
}))
vi.mock('@/themes/context', () => ({ useTheme: () => ({ resolvedMode: 'dark' }) }))
// Only canvas painting is replaced; owner resolution, RPC dispatch, effects,
// cancellation, and the actual FloatingPet mount all run unchanged.
vi.mock('./pet-sprite', () => ({
  PetSprite: ({ info }: { info: PetInfo }) => <canvas aria-label={`${info.displayName} pet`} />,
  roamWalkRow: () => ({ mirror: false, row: undefined })
}))

function pet(owner: string): PetInfo {
  return {
    enabled: true,
    slug: `${owner}-pet`,
    displayName: owner,
    spritesheetBase64: owner,
    spritesheetRevision: `${owner}:1`,
    scale: 0.33
  }
}

function chat(profile: string): SessionInfo {
  return {
    id: `${profile}-chat`,
    profile,
    connection_id: 'local',
    ended_at: null,
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 0,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: 0,
    title: 'Bot Chat',
    tool_call_count: 0
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void

  const promise = new Promise<T>((done, fail) => {
    resolve = done
    reject = fail
  })

  return { promise, resolve, reject }
}

async function mount() {
  await act(async () => {
    render(
      <MemoryRouter>
        <FloatingPet />
      </MemoryRouter>
    )
  })
}

beforeEach(() => {
  vi.useFakeTimers()
  rpc.mockReset()
  rpc.mockImplementation(async (_method: string, params: { profile: string }) => pet(params.profile))
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $activeGatewayProfile.set('A')
  $sessions.set(['A', 'B', 'C'].map(chat))
  $gatewayState.set('open')
  $changeEventsAvailable.set(true)
  $petChange.set({ tick: 0 })
  $petOverlayActive.set(false)
  resetPetGallery()
  setPetInfo({ enabled: false })
})

afterEach(() => {
  cleanup()
  resetPetGallery()
  vi.clearAllTimers()
  vi.useRealTimers()
  $gatewayState.set('closed')
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $sessions.set([])
  $activeGatewayProfile.set('default')
  $petChange.set({ tick: 0 })
  $changeEventsAvailable.set(false)
  setPetInfo({ enabled: false })
})

describe('FloatingPet owner lifecycle', () => {
  it('keeps a preloaded mascot and gallery on first mount while metadata is stalled', async () => {
    const pending = deferred<PetInfo>()
    rpc.mockReturnValue(pending.promise)
    setPetInfo(pet('A'))
    const gallery = { enabled: true, active: 'A-pet', pets: [] }
    $petGallery.set(gallery)
    await mount()
    expect(screen.queryByLabelText('A pet')).not.toBeNull()
    expect($petGallery.get()).toBe(gallery)
  })

  it('switches A to B to C with a fixed launch profile and drops the delayed B reply', async () => {
    const b = deferred<PetInfo>()
    rpc.mockImplementation(async (_method: string, params: { profile: string }) =>
      params.profile === 'B' ? b.promise : pet(params.profile)
    )
    $selectedStoredSessionId.set('A-chat')
    await mount()
    expect(screen.queryByLabelText('A pet')).not.toBeNull()
    await act(async () => {
      $selectedStoredSessionId.set('B-chat')
    })
    expect(screen.queryByLabelText('A pet')).toBeNull()
    expect(rpc.mock.calls.some(([, params]) => params.profile === 'B')).toBe(true)
    await act(async () => {
      $selectedStoredSessionId.set('C-chat')
    })
    expect(screen.queryByLabelText('C pet')).not.toBeNull()
    await act(async () => {
      b.resolve(pet('B'))
    })
    expect(screen.queryByLabelText('B pet')).toBeNull()
    expect(screen.queryByLabelText('C pet')).not.toBeNull()
    expect($activeGatewayProfile.get()).toBe('A')
  })

  it('clears the launch mascot immediately on a disable event despite stalled reads', async () => {
    const pending = deferred<PetInfo>()
    rpc.mockReturnValue(pending.promise)
    setPetInfo(pet('A'))
    await mount()
    expect(screen.queryByLabelText('A pet')).not.toBeNull()
    const before = rpc.mock.calls.length
    await act(async () => {
      notifyPetChanged({ enabled: false })
    })
    expect(screen.queryByLabelText('A pet')).toBeNull()
    expect(rpc.mock.calls.length).toBe(before)
    await act(async () => {
      pending.resolve(pet('A'))
    })
    expect(screen.queryByLabelText('A pet')).toBeNull()
  })

  it('does not apply a launch disable to B, but clears when B own metadata is disabled', async () => {
    const b = deferred<PetInfo>()
    $selectedStoredSessionId.set('B-chat')
    setPetInfo(pet('B'))
    rpc.mockReturnValue(b.promise)
    await mount()
    await act(async () => {
      notifyPetChanged({ enabled: false })
    })
    expect(screen.queryByLabelText('B pet')).not.toBeNull()
    await act(async () => {
      b.resolve({ enabled: false })
    })
    expect(screen.queryByLabelText('B pet')).toBeNull()
  })

  it('does not issue a fallback full read after an old metadata request is cancelled', async () => {
    const b = deferred<PetInfo>()
    $selectedStoredSessionId.set('B-chat')
    setPetInfo(pet('B'))
    rpc.mockImplementation(async (method: string, params: { profile: string }) =>
      params.profile === 'B' && method === 'pet.info.meta' ? b.promise : pet(params.profile)
    )
    await mount()
    await act(async () => {
      $selectedStoredSessionId.set('C-chat')
    })
    await act(async () => {
      b.reject(new Error('old transport failed'))
    })
    expect(rpc.mock.calls.filter(([method, params]) => method === 'pet.info' && params.profile === 'B')).toHaveLength(0)
    expect(screen.queryByLabelText('C pet')).not.toBeNull()
  })
})
