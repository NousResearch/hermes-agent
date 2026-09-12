import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const gatewayMocks = vi.hoisted(() => {
  const instances: Array<{
    connectionState: string
    emit: (event: { payload?: Record<string, unknown>; session_id?: string; type: string }) => void
    request: ReturnType<typeof vi.fn>
  }> = []

  return { instances }
})

vi.mock('@/hermes', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  setApiRequestConnection: vi.fn(),
  HermesGateway: class {
    connectionState = 'closed'
    private eventHandlers = new Set<
      (event: { payload?: Record<string, unknown>; session_id?: string; type: string }) => void
    >()
    request = vi.fn(async (method: string, params: Record<string, unknown>) => ({ method, params }))

    constructor() {
      gatewayMocks.instances.push(this as never)
    }

    connect = async (): Promise<void> => {
      this.connectionState = 'open'
    }

    close = (): void => {
      this.connectionState = 'closed'
    }

    onEvent = (
      handler: (event: { payload?: Record<string, unknown>; session_id?: string; type: string }) => void
    ): (() => void) => {
      this.eventHandlers.add(handler)

      return () => this.eventHandlers.delete(handler)
    }

    onState = (): (() => void) => () => undefined

    emit = (event: { payload?: Record<string, unknown>; session_id?: string; type: string }): void => {
      for (const handler of this.eventHandlers) {
        handler(event)
      }
    }
  }
}))

const {
  closeSecondaryGateways,
  configureGatewayRegistry,
  ensureGatewayForProfile,
  retireLocalProfileGateways,
  setPrimaryGateway
} = await import('./gateway')

const { $profiles } = await import('./profile')

const { $activeSessionId, _resetSessionOwnerHintsForTests, setSessionOwnerHint, setSessions } =
  await import('./session')

const { $gateway } = await import('./gateway')
const { clearAllPrompts } = await import('./prompts')

const {
  $sessionStates,
  $sessionTiles,
  clearAllSessionStates,
  forgetProfileOnlyRuntimeOwners,
  knownOwnerForSession,
  recordSessionEventScope,
  requestForOwnedSession
} = await import('./session-states')

const { isSessionOwnerResolutionError } = await import('./session-owner-resolution')

function installDesktop(): void {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    getConnection: vi.fn(async (profile: string) => ({
      authMode: 'token',
      mode: 'local',
      profile,
      token: 'test-token',
      wsUrl: `wss://${profile}.invalid/ws`
    })),
    notify: vi.fn()
  }
}

beforeEach(() => {
  installDesktop()
  gatewayMocks.instances.length = 0
  $profiles.set([{ name: 'default' }, { name: 'research' }] as never)
  $sessionTiles.set([])
  setSessions([])
  clearAllSessionStates()
  clearAllPrompts()
  $activeSessionId.set(null)
  _resetSessionOwnerHintsForTests({ storage: true })

  // This is the exact event fan-in installed by useGatewayBoot: the secondary
  // producer calls config.onEvent, which records scope before UI dispatch.
  configureGatewayRegistry({
    onLocalProfileRetired: forgetProfileOnlyRuntimeOwners,
    onEvent: event => {
      recordSessionEventScope(event)
    }
  })
  const primary = { connectionState: 'open', request: vi.fn() }
  $gateway.set(primary as never)
  setPrimaryGateway(primary as never, 'default')
})

afterEach(() => {
  closeSecondaryGateways()
  clearAllSessionStates()
  $sessionTiles.set([])
  $profiles.set([])
  setSessions([])
  _resetSessionOwnerHintsForTests({ storage: true })
  clearAllPrompts()
  $activeSessionId.set(null)
  $gateway.set(null)
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

describe('profile-only secondary approval ownership', () => {
  it('keeps a profile-only secondary event owner across a focus switch and dispatches approval on that secondary', async () => {
    await expect(ensureGatewayForProfile('research')).resolves.toBeUndefined()
    const secondary = gatewayMocks.instances[0]

    expect(secondary).toBeTruthy()
    // No tile, row, hint, or runtime→stored state mirror is available.
    expect($sessionTiles.get()).toEqual([])
    expect($sessionStates.get()).toEqual({})

    secondary.emit({ session_id: 'rt-secondary', type: 'approval.request' })
    await expect(ensureGatewayForProfile('default')).resolves.toBeUndefined()

    expect(knownOwnerForSession('rt-secondary')).toBe('research')

    const ambient = vi.fn(async () => ({ via: 'ambient' }))
    await expect(
      requestForOwnedSession('rt-secondary', ambient as never, 'approval.respond', {
        choice: 'once',
        session_id: 'rt-secondary'
      })
    ).resolves.toEqual({
      method: 'approval.respond',
      params: { choice: 'once', session_id: 'rt-secondary' }
    })

    expect(secondary.request).toHaveBeenCalledWith('approval.respond', {
      choice: 'once',
      session_id: 'rt-secondary'
    })
    expect(ambient).not.toHaveBeenCalled()
  })

  it('rejects unproven profiles and retires only transient local ownership, never durable exact routes', async () => {
    recordSessionEventScope({ profile: 'research', session_id: 'rt-unproven' })
    expect(knownOwnerForSession('rt-unproven')).toBeUndefined()
    await expect(requestForOwnedSession('rt-unproven', vi.fn() as never, 'approval.respond', {})).rejects.toSatisfy(
      isSessionOwnerResolutionError
    )

    await ensureGatewayForProfile('research')
    const secondary = gatewayMocks.instances[0]
    secondary.emit({ session_id: 'rt-retired', type: 'approval.request' })
    secondary.emit({ session_id: 'rt-durable', type: 'approval.request' })
    expect(knownOwnerForSession('rt-retired')).toBe('research')
    const exact = { connectionId: 'remote-same-name', profile: 'research' }
    setSessionOwnerHint('rt-durable', exact)
    recordSessionEventScope({ ...exact, session_id: 'rt-remote' })
    expect(knownOwnerForSession('rt-durable')).toEqual(exact)
    retireLocalProfileGateways('research')
    expect(knownOwnerForSession('rt-retired')).toBeUndefined()
    expect(knownOwnerForSession('rt-durable')).toEqual(exact)
    expect(knownOwnerForSession('rt-remote')).toEqual(exact)
    const getConnection = window.hermesDesktop!.getConnection
    vi.mocked(getConnection).mockClear()
    await expect(requestForOwnedSession('rt-retired', vi.fn() as never, 'approval.respond', {})).rejects.toSatisfy(
      isSessionOwnerResolutionError
    )
    expect(getConnection).not.toHaveBeenCalled()
  })
})
