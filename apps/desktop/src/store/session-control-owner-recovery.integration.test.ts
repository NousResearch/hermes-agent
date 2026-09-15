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
  $gateway,
  closeSecondaryGateways,
  configureGatewayRegistry,
  ensureGatewayForProfile,
  setPrimaryGateway
} = await import('./gateway')

const { $profiles } = await import('./profile')

const { $activeSessionId, _resetSessionOwnerHintsForTests, setSessions } = await import('./session')

const { clearAllPrompts } = await import('./prompts')

const { $sessionTiles, clearAllSessionStates, forgetProfileOnlyRuntimeOwners, recordSessionEventScope } =
  await import('./session-states')

const {
  $sessionControlBySession,
  clearAllSessionControl,
  refreshSessionControl,
  setSessionControlOwnerProbe
} = await import('./session-control')

/** A valid backend control payload: parseSessionControlSnapshot is exact-shape. */
const CONTROL_SNAPSHOT = {
  goal: null,
  heartbeat: null,
  loop: null,
  revision: 'rev-ownerless',
  updated_at: 1_700_000_000
}

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

let primary: { connectionState: string; request: ReturnType<typeof vi.fn> }

beforeEach(() => {
  installDesktop()
  gatewayMocks.instances.length = 0
  // More than one profile: the ambient gateway is NOT the owner by
  // construction, so an unresolvable runtime id must fail closed.
  $profiles.set([{ name: 'default' }, { name: 'research' }] as never)
  $sessionTiles.set([])
  setSessions([])
  clearAllSessionStates()
  clearAllSessionControl()
  clearAllPrompts()
  $activeSessionId.set(null)
  _resetSessionOwnerHintsForTests({ storage: true })
  setSessionControlOwnerProbe(null)

  configureGatewayRegistry({
    onLocalProfileRetired: forgetProfileOnlyRuntimeOwners,
    onEvent: event => {
      recordSessionEventScope(event)
    }
  })

  primary = { connectionState: 'open', request: vi.fn(async () => ({ control: CONTROL_SNAPSHOT })) }
  $gateway.set(primary as never)
  setPrimaryGateway(primary as never, 'default')
})

afterEach(() => {
  closeSecondaryGateways()
  clearAllSessionStates()
  clearAllSessionControl()
  setSessionControlOwnerProbe(null)
  $sessionTiles.set([])
  $profiles.set([])
  setSessions([])
  _resetSessionOwnerHintsForTests({ storage: true })
  clearAllPrompts()
  $activeSessionId.set(null)
  $gateway.set(null)
  delete (window as { hermesDesktop?: unknown }).hermesDesktop
})

describe('session-control ownerless runtime recovery (#107502)', () => {
  it('re-resolves the owner through the probe and retries the read on that backend', async () => {
    await ensureGatewayForProfile('research')
    const secondary = gatewayMocks.instances[0]
    secondary.request.mockResolvedValue({ control: CONTROL_SNAPSHOT })

    const probe = vi.fn(async () => 'research')
    setSessionControlOwnerProbe(probe)

    const entry = await refreshSessionControl('rt-ownerless')

    expect(probe).toHaveBeenCalledTimes(1)
    expect(probe).toHaveBeenCalledWith('rt-ownerless')
    expect(secondary.request).toHaveBeenCalledWith('session.control.read', { session_id: 'rt-ownerless' })
    expect(primary.request).not.toHaveBeenCalled()
    expect(entry).toMatchObject({ capability: 'supported', error: null, loading: false, pendingAction: null })
    expect(entry!.snapshot!.revision).toBe('rev-ownerless')
  })

  it('keeps an unrecoverable ownerless read quiet instead of surfacing the routing error', async () => {
    const probe = vi.fn(async () => undefined)
    setSessionControlOwnerProbe(probe)

    const entry = await refreshSessionControl('rt-ownerless')

    expect(probe).toHaveBeenCalledWith('rt-ownerless')
    expect(entry).toMatchObject({ capability: 'unknown', error: null, loading: false, snapshot: null })
    expect(primary.request).not.toHaveBeenCalled()
    expect(gatewayMocks.instances).toHaveLength(0)
  })

  it('never routes an ownerless read to the ambient gateway (fail-closed stays)', async () => {
    await refreshSessionControl('rt-ownerless')

    expect(primary.request).not.toHaveBeenCalled()
  })

  it('does not consult the probe when the owner ladder already names an owner', async () => {
    $profiles.set([{ name: 'default' }] as never)

    const probe = vi.fn(async () => 'research')
    setSessionControlOwnerProbe(probe)

    await refreshSessionControl('rt-legacy')

    expect(probe).not.toHaveBeenCalled()
    expect(primary.request).toHaveBeenCalledWith('session.control.read', { session_id: 'rt-legacy' })
  })

  it('still publishes an ordinary backend failure for a routable session', async () => {
    $profiles.set([{ name: 'default' }] as never)
    primary.request.mockRejectedValue(new Error('backend unavailable'))

    const entry = await refreshSessionControl('rt-legacy')

    expect(entry!.error).toBe('backend unavailable')
    expect(entry!.loading).toBe(false)
  })

  it('still reports a genuine owner-resolution failure for a user-initiated action', async () => {
    const { runSessionControlAction } = await import('./session-control')

    await expect(runSessionControlAction('rt-ownerless', 'goal.pause')).rejects.toMatchObject({
      name: 'SessionOwnerResolutionError'
    })

    expect($sessionControlBySession.get()['rt-ownerless']!.error).toContain('(session.control)')
  })
})
