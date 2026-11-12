// The wake word's profile switch pins the new-chat intent mid-swap so the wake
// draft's first send lands in the target even before the socket settles
// (#79005). But a wake is ambient — nobody picked a profile — so that pin must
// not outlive the dial the way a deliberate pick's does (#149779): it would
// steer every later "+"/⌘T session in the window into the woken profile until
// some other action rewrote it.
//
// Each test dials a UNIQUE profile name: the swap machinery and the
// $activeGatewayProfile mirror are module state with no reset hook, and a
// settled dial from a previous test would fast-path the next one.
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as Hermes from '@/hermes'

import { deferred } from '../test/deferred'

const gatewayMocks = vi.hoisted(() => {
  const instances: { close: ReturnType<typeof vi.fn>; connectionState: string }[] = []

  return {
    connect: vi.fn(async (_wsUrl: string): Promise<void> => undefined),
    instances
  }
})

vi.mock('@/hermes', async importOriginal => {
  const actual = await importOriginal<typeof Hermes>()

  return {
    ...actual,
    getProfiles: vi.fn(async () => ({ profiles: [] })),
    hermesApi: vi.fn(async () => ({ current: 'default', profiles: [] })),
    HermesGateway: class {
      connectionState = 'closed'
      wsUrl = ''
      connect = async (url: string): Promise<void> => {
        this.wsUrl = url
        await gatewayMocks.connect(url)
        this.connectionState = 'open'
      }
      close = () => {
        this.connectionState = 'closed'
      }
      onEvent = () => () => undefined
      onState = () => () => undefined
      request = async () => ({})
      constructor() {
        gatewayMocks.instances.push(this as never)
      }
    }
  }
})
vi.mock('@/store/session', () => ({
  $connection: atom(null),
  clearComposerSelectionOwner: vi.fn(),
  setComposerSelectionOwner: vi.fn(),
  setConnection: vi.fn(),
  setGatewayState: vi.fn()
}))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))
vi.mock('@/store/profile-remote-override', () => ({ notifyRemoteOverrideAuthFailure: vi.fn(() => false) }))
vi.mock('@/lib/query-client', () => ({ invalidateProfileScopedQueries: vi.fn() }))

const { $newChatProfile, pinNewChatProfile, wakeSessionInProfile } = await import('./profile')

function installDesktop(stub: Record<string, unknown>): void {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = stub
}

function descriptorFor(profile: string) {
  return {
    authMode: 'token',
    baseUrl: `https://${profile}.invalid`,
    mode: 'local',
    profile,
    token: 'fake-test-token',
    wsUrl: `wss://${profile}.invalid/ws`
  }
}

function stubDescriptorLookup(): void {
  installDesktop({ getConnection: vi.fn(async ({ profile }: { profile: string }) => descriptorFor(profile)) })
}

beforeEach(() => {
  gatewayMocks.instances.length = 0
  $newChatProfile.set(null)
})

afterEach(() => {
  vi.clearAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('wakeSessionInProfile — the ambient wake pin is scoped to the swap (#149779)', () => {
  it('drops the pin once the dial settles, so later "+" sessions resolve from the window state again', async () => {
    stubDescriptorLookup()

    await wakeSessionInProfile('wake-a')

    expect($newChatProfile.get()).toBeNull()
  })

  it('keeps the pin when the dial fails — the wake draft still targets the failed profile', async () => {
    gatewayMocks.connect.mockRejectedValue(new Error('backend unreachable'))
    stubDescriptorLookup()

    // The failure is surfaced (#81094) instead of rejecting out of the handler.
    await expect(wakeSessionInProfile('wake-b')).resolves.toBeUndefined()

    expect($newChatProfile.get()).toBe('wake-b')
  })

  it('does not clobber a deliberate pick made while the dial was in flight', async () => {
    const gate = deferred<void>()
    gatewayMocks.connect.mockImplementation(async () => {
      await gate.promise
    })
    stubDescriptorLookup()

    const settled = wakeSessionInProfile('wake-c')

    // Mid-swap the wake pin is live — the draft's send must land in 'wake-c'.
    expect($newChatProfile.get()).toBe('wake-c')

    // A deliberate pick (rail click, per-profile "+") lands while the dial runs.
    pinNewChatProfile('home')

    gate.resolve()
    await settled

    expect($newChatProfile.get()).toBe('home')
  })
})
