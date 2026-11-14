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
import { notifyError } from '@/store/notifications'

import { deferred } from '../test/deferred'

import { closeSecondaryGateways } from './gateway'

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
  // resolveConnectionForProfile passes the profile name directly.
  installDesktop({ getConnection: vi.fn(async (profile: string) => descriptorFor(profile)) })
}

beforeEach(() => {
  // mockReset also restores the vi.fn factory implementation — a rejected or
  // gated connect from an earlier test must not poison the next one.
  gatewayMocks.connect.mockReset()
  gatewayMocks.instances.length = 0
  $newChatProfile.set(null)
})

afterEach(() => {
  vi.clearAllMocks()
  // A failed dial arms a real reconnect timer and a wantOpen secondary; drop
  // them so no background re-dial races the next test's mock.
  closeSecondaryGateways()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('wakeSessionInProfile — the ambient wake pin is scoped to the swap (#149779)', () => {
  it('dials the target and drops the pin once the dial settles', async () => {
    stubDescriptorLookup()

    await wakeSessionInProfile('wake-a')

    expect(gatewayMocks.connect).toHaveBeenCalledWith(expect.stringContaining('wake-a.invalid'))
    expect($newChatProfile.get()).toBeNull()
  })

  it('keeps the pin when the dial fails and surfaces the failure', async () => {
    gatewayMocks.connect.mockRejectedValue(new Error('backend unreachable'))
    stubDescriptorLookup()

    // The failure is surfaced (#81094) instead of rejecting out of the handler.
    await expect(wakeSessionInProfile('wake-b')).resolves.toBeUndefined()

    expect(gatewayMocks.connect).toHaveBeenCalledWith(expect.stringContaining('wake-b.invalid'))
    expect(notifyError).toHaveBeenCalledTimes(1)
    expect($newChatProfile.get()).toBe('wake-b')
  })

  it('does not clobber a deliberate pick of another profile made mid-dial', async () => {
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

    expect(gatewayMocks.connect).toHaveBeenCalledWith(expect.stringContaining('wake-c.invalid'))
    expect($newChatProfile.get()).toBe('home')
  })

  it('does not clobber a deliberate re-pin of the same profile made mid-dial', async () => {
    const gate = deferred<void>()
    gatewayMocks.connect.mockImplementation(async () => {
      await gate.promise
    })
    stubDescriptorLookup()

    const settled = wakeSessionInProfile('wake-d')
    expect($newChatProfile.get()).toBe('wake-d')

    // A deliberate pick of the SAME profile mid-dial re-pins with identical
    // values — only the intent revision can tell it from the ambient pin.
    pinNewChatProfile('wake-d')

    gate.resolve()
    await settled

    expect($newChatProfile.get()).toBe('wake-d')
  })
})
