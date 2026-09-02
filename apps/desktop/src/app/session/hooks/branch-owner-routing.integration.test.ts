/**
 * End-to-end owner routing for BRANCH (the #97764-adjacent strand).
 *
 * The unit tests in use-session-actions.test.tsx mock `@/store/gateway`, so
 * they prove the branch path ASKS for the right route. They cannot prove the
 * routing layer HONOURS it. This file mocks nothing inside the router: the real
 * `requestGatewayForAgent` runs against a fake Electron bridge + transport, so
 * a regression that re-collapses a registry route onto the ambient socket fails
 * here even if the call-site assertions still pass.
 *
 * Reproduces the reported shape: a session owned by a remote connection
 * ("pandora") is branched while a different backend is active. Before the fix
 * the create rode the ambient socket and the child was created on the wrong
 * backend (or nowhere), stranding an optimistic sidebar row on an id no backend
 * owned — "Couldn't load this session".
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'

// Every socket the registry dials, and every RPC that travelled over one.
const dialed: { connectionId: string; profile: string }[] = []
const sent: { method: string; params: Record<string, unknown>; url: string }[] = []

class FakeHermesGateway {
  connectionState = 'closed'
  private url = ''

  async connect(wsUrl: string) {
    if (typeof wsUrl !== 'string' || !wsUrl.startsWith('ws')) {
      throw new Error(`bad ws url: ${String(wsUrl)}`)
    }

    this.url = wsUrl
    this.connectionState = 'open'
  }

  async request<T>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    sent.push({ method, params, url: this.url })

    if (method === 'session.create' || method === 'session.branch') {
      return { session_id: 'branch-runtime', stored_session_id: 'branch-stored' } as T
    }

    return {} as T
  }

  close() {
    this.connectionState = 'closed'
  }

  onEvent(_listener: (event: unknown) => void) {
    return () => undefined
  }

  onState(_listener: (state: unknown) => void) {
    return () => undefined
  }

  onStateChange(_listener: (state: unknown) => void) {
    return () => undefined
  }

  on() {}
  off() {}
  addEventListener() {}
  removeEventListener() {}
}

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  HermesGateway: FakeHermesGateway,
  setApiRequestConnection: vi.fn()
}))

describe('branch owner routing (real router, faked transport)', () => {
  beforeEach(() => {
    dialed.length = 0
    sent.length = 0
    vi.resetModules()

    // A registry with two backends exposing the SAME profile name — the exact
    // ambiguity that makes profile-only routing wrong.
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      getConnection: async () => ({ mode: 'local' }),
      getConnectionFor: async ({ connectionId, profile }: { connectionId: string; profile: string }) => {
        dialed.push({ connectionId, profile })

        return { connectionId, mode: 'remote', profile }
      },
      getGatewayWsUrlFor: async ({ connectionId, profile }: { connectionId: string; profile: string }) =>
        `ws://${connectionId}/gateway?profile=${profile}`,
      touchBackend: async () => undefined
    }
  })

  it('dials the parent connection and sends the create over that socket', async () => {
    const { requestGatewayForAgent } = await import('@/store/gateway')

    await requestGatewayForAgent('pandora', 'default', 'session.create', {
      parent_session_id: 'stored-parent',
      source: 'desktop'
    })

    // The registry resolved a socket for the PARENT's connection...
    expect(dialed).toContainEqual({ connectionId: 'pandora', profile: 'default' })

    // ...and the create actually travelled over that socket.
    const create = sent.find(entry => entry.method === 'session.create')
    expect(create).toBeDefined()
    expect(create!.url).toContain('pandora')
    expect(create!.params).toMatchObject({ parent_session_id: 'stored-parent' })
  })

  it('keeps two same-named profiles on separate sockets', async () => {
    const { requestGatewayForAgent } = await import('@/store/gateway')

    await requestGatewayForAgent('pandora', 'default', 'session.create', { source: 'desktop' })
    await requestGatewayForAgent('other-box', 'default', 'session.create', { source: 'desktop' })

    const urls = sent.filter(entry => entry.method === 'session.create').map(entry => entry.url)

    expect(urls).toHaveLength(2)
    // Same profile name, different backends — they must NOT share a socket.
    expect(new Set(urls).size).toBe(2)
    expect(urls.some(url => url.includes('pandora'))).toBe(true)
    expect(urls.some(url => url.includes('other-box'))).toBe(true)
  })
})
