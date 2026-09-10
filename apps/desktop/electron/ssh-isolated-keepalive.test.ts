/**
 * #106935: Desktop main must hold a long-lived keep-alive WebSocket for every
 * published SSH-isolated backend. Idle-exit (#101626) treats accepted WS as
 * ownership liveness; renderer sockets can drop while sshConnections still owns
 * the scope. Sticky artifacts (nonce / token file / lockfile) are NOT liveness.
 *
 * Wiring through main.ts is asserted via pool-stop / bootstrap-coordinator
 * behavior tests — AGENTS.md forbids reading `.ts` source from tests.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

function makeFakeWs(): { FakeWs: new (url: string) => any; instances: any[] } {
  const instances: any[] = []

  class FakeWs {
    url: string
    closed = false
    listeners: Record<string, Array<(event?: any) => void>> = {}

    constructor(url: string) {
      this.url = url
      instances.push(this)
    }

    addEventListener(type: string, fn: (event?: any) => void) {
      ;(this.listeners[type] ||= []).push(fn)
    }

    close() {
      this.closed = true
    }

    emit(type: string, event?: any) {
      for (const fn of this.listeners[type] || []) {
        fn(event)
      }
    }
  }

  return { FakeWs, instances }
}

describe('ssh-isolated keep-alive registry (#106935)', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('holds an open WebSocket for the published baseUrl+token until stop', async () => {
    const { createSshIsolatedKeepaliveRegistry } = await import('./ssh-isolated-keepalive')
    const { FakeWs, instances } = makeFakeWs()
    const registry = createSshIsolatedKeepaliveRegistry({ WebSocketImpl: FakeWs })

    registry.start('conn:office::work', {
      baseUrl: 'http://127.0.0.1:53101',
      token: 'sess-work'
    })

    expect(instances).toHaveLength(1)
    expect(instances[0].url).toBe('ws://127.0.0.1:53101/api/ws?token=sess-work')
    expect(instances[0].closed).toBe(false)
    expect(registry.isArmed('conn:office::work')).toBe(true)

    instances[0].emit('open')
    expect(instances[0].closed).toBe(false)
    expect(registry.openUrl('conn:office::work')).toBe('ws://127.0.0.1:53101/api/ws?token=sess-work')

    registry.stop('conn:office::work')
    expect(instances[0].closed).toBe(true)
    expect(registry.isArmed('conn:office::work')).toBe(false)
    expect(registry.openUrl('conn:office::work')).toBeNull()
  })

  it('stop clears the keep-alive and does not leak a later reconnect', async () => {
    const { createSshIsolatedKeepaliveRegistry } = await import('./ssh-isolated-keepalive')
    const { FakeWs, instances } = makeFakeWs()
    const registry = createSshIsolatedKeepaliveRegistry({
      WebSocketImpl: FakeWs,
      reconnectDelayMs: 25
    })

    registry.start('scope-a', { baseUrl: 'http://127.0.0.1:9', token: 'tok' })
    expect(instances).toHaveLength(1)

    instances[0].emit('open')
    registry.stop('scope-a')
    expect(instances[0].closed).toBe(true)

    instances[0].emit('close', { code: 1006 })
    await new Promise(resolve => setTimeout(resolve, 50))
    expect(instances).toHaveLength(1)
    expect(registry.isArmed('scope-a')).toBe(false)
  })

  it('replaces a prior socket when the same scope is published again', async () => {
    const { createSshIsolatedKeepaliveRegistry } = await import('./ssh-isolated-keepalive')
    const { FakeWs, instances } = makeFakeWs()
    const registry = createSshIsolatedKeepaliveRegistry({ WebSocketImpl: FakeWs })

    registry.start('scope-a', { baseUrl: 'http://127.0.0.1:9', token: 'old' })
    registry.start('scope-a', { baseUrl: 'http://127.0.0.1:9', token: 'new' })

    expect(instances).toHaveLength(2)
    expect(instances[0].closed).toBe(true)
    expect(instances[1].closed).toBe(false)
    expect(instances[1].url).toBe('ws://127.0.0.1:9/api/ws?token=new')
  })

  it('fail-open: a missing WebSocket impl does not throw and does not invent liveness', async () => {
    const { createSshIsolatedKeepaliveRegistry } = await import('./ssh-isolated-keepalive')
    const registry = createSshIsolatedKeepaliveRegistry({ WebSocketImpl: undefined })

    expect(() => {
      registry.start('scope-a', { baseUrl: 'http://127.0.0.1:9', token: 'tok' })
    }).not.toThrow()
    expect(registry.isArmed('scope-a')).toBe(false)
    expect(registry.openUrl('scope-a')).toBeNull()
  })

  it('treats empty-string scope as the v1/global SSH primary and holds a WS until stop', async () => {
    const { createSshIsolatedKeepaliveRegistry } = await import('./ssh-isolated-keepalive')
    const { FakeWs, instances } = makeFakeWs()
    const registry = createSshIsolatedKeepaliveRegistry({ WebSocketImpl: FakeWs })

    registry.start('', { baseUrl: 'http://127.0.0.1:53100', token: 'primary' })

    expect(instances).toHaveLength(1)
    expect(instances[0].url).toBe('ws://127.0.0.1:53100/api/ws?token=primary')
    expect(instances[0].closed).toBe(false)
    expect(registry.isArmed('')).toBe(true)

    instances[0].emit('open')
    expect(registry.openUrl('')).toBe('ws://127.0.0.1:53100/api/ws?token=primary')

    registry.stop('')
    expect(instances[0].closed).toBe(true)
    expect(registry.isArmed('')).toBe(false)
    expect(registry.openUrl('')).toBeNull()
  })

  it('rejects missing baseUrl or token even when the scope is the empty primary key', async () => {
    const { createSshIsolatedKeepaliveRegistry } = await import('./ssh-isolated-keepalive')
    const { FakeWs, instances } = makeFakeWs()
    const registry = createSshIsolatedKeepaliveRegistry({ WebSocketImpl: FakeWs })

    registry.start('', { baseUrl: '', token: 'tok' })
    registry.start('', { baseUrl: 'http://127.0.0.1:9', token: '' })
    registry.start('', { baseUrl: 'http://127.0.0.1:9', token: undefined as unknown as string })
    registry.start('', { baseUrl: undefined as unknown as string, token: 'tok' })

    expect(instances).toHaveLength(0)
    expect(registry.isArmed('')).toBe(false)
  })

  it('tracks independent scopes so tearing down one sibling cannot drop the other', async () => {
    const { createSshIsolatedKeepaliveRegistry } = await import('./ssh-isolated-keepalive')
    const { FakeWs, instances } = makeFakeWs()
    const registry = createSshIsolatedKeepaliveRegistry({ WebSocketImpl: FakeWs })

    registry.start('profile-less', { baseUrl: 'http://127.0.0.1:53101', token: 'a' })
    registry.start('profile-work', { baseUrl: 'http://127.0.0.1:53102', token: 'b' })

    expect(instances).toHaveLength(2)
    registry.stop('profile-work')
    expect(instances[0].closed).toBe(false)
    expect(instances[1].closed).toBe(true)
    expect(registry.isArmed('profile-less')).toBe(true)
  })
})
