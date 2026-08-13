import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The global-remote share (backend routing case 3): every profile is served
// by the PRIMARY backend over one host, and getConnection() tags the shared
// descriptor with `profile`. Dialing a second WebSocket at that descriptor
// used to fail over SSH (per-backend tunnel/ticket) and poison the active
// gateway with a closed socket — "Hermes gateway is not connected" for every
// profile except the primary. These tests pin the fix: a profile routed to
// the shared primary activates the primary socket instead of dialing.

vi.mock('@/hermes', () => ({
  HermesGateway: class {
    connectionState = 'closed'
    connect = vi.fn(async () => {
      throw new Error('dialed a socket for a shared-primary profile')
    })
    onEvent = vi.fn(() => () => {})
    onState = vi.fn(() => () => {})
  }
}))
vi.mock('@/store/session', () => ({ setGatewayState: vi.fn() }))
vi.mock('@/store/notify-baseline', () => ({ markNativeNotifyBaseline: vi.fn() }))

const { $gateway, configureGatewayRegistry, ensureGatewayForProfile, setPrimaryGateway } = await import('./gateway')

type DesktopStub = { getConnection: ReturnType<typeof vi.fn> }

function installDesktop(stub: DesktopStub): void {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = stub
}

function makePrimary(): { connectionState: string } {
  // Only connectionState is consulted by setActive/isOpen for these paths.
  return { connectionState: 'open' }
}

beforeEach(() => {
  configureGatewayRegistry({
    onEvent: vi.fn(),
    primaryProfile: 'default'
  } as never)
})

afterEach(() => {
  vi.clearAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('ensureGatewayForProfile under a shared global remote', () => {
  it('activates the primary socket for a profile tagged onto the shared descriptor', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop({
      // Shared descriptor: primary connection tagged with the profile.
      getConnection: vi.fn(async () => ({ port: 4242, profile: 'venture', token: 't' }))
    })

    await ensureGatewayForProfile('venture')

    expect($gateway.get()).toBe(primary)
  })

  it('still pools a socket for profiles with their own descriptor (untagged)', async () => {
    const primary = makePrimary()
    setPrimaryGateway(primary as never, 'default')
    installDesktop({
      // Own descriptor: no profile tag → normal pooled path (dial attempted).
      getConnection: vi.fn(async () => ({ port: 5151, token: 't2' }))
    })

    await ensureGatewayForProfile('worker')

    // The pooled path dialed (our stub throws, so the socket stays closed and
    // reconnect is scheduled) — the important part is it did NOT silently
    // reuse the primary.
    expect($gateway.get()).not.toBe(primary)
  })
})
