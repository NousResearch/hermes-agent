/**
 * #106935: Desktop main must hold a long-lived keep-alive WebSocket for every
 * published SSH-isolated backend. Idle-exit (#101626) treats accepted WS as
 * ownership liveness; renderer sockets can drop while sshConnections still owns
 * the scope. Sticky artifacts (nonce / token file / lockfile) are NOT liveness.
 *
 * The pool-stop teardown fence that closes this socket is asserted in
 * pool-stop.test.ts (afterStop) — AGENTS.md forbids reading `.ts` source from tests.
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

  it('holds an open WebSocket per scope until that scope stops, then never reconnects it', async () => {
    const { createSshIsolatedKeepaliveRegistry } = await import('./ssh-isolated-keepalive')
    const { FakeWs, instances } = makeFakeWs()
    const registry = createSshIsolatedKeepaliveRegistry({
      WebSocketImpl: FakeWs,
      reconnectDelayMs: 25
    })

    registry.start('conn:office::work', {
      baseUrl: 'http://127.0.0.1:53101',
      token: 'sess-work'
    })
    registry.start('conn:office::less', { baseUrl: 'http://127.0.0.1:53102', token: 'sess-less' })

    expect(instances).toHaveLength(2)
    expect(instances[0].url).toBe('ws://127.0.0.1:53101/api/ws?token=sess-work')
    expect(registry.isArmed('conn:office::work')).toBe(true)

    instances[0].emit('open')
    expect(registry.openUrl('conn:office::work')).toBe('ws://127.0.0.1:53101/api/ws?token=sess-work')

    // A dropped socket is redialled with backoff (25 → 50 ms), so a dead tunnel is not
    // hammered every interval until the scope is torn down.
    vi.useFakeTimers()
    instances[1].emit('close', { code: 1006 })
    await vi.advanceTimersByTimeAsync(25)
    expect(instances).toHaveLength(3)
    instances[2].emit('close', { code: 1006 })
    await vi.advanceTimersByTimeAsync(25)
    expect(instances).toHaveLength(3)
    await vi.advanceTimersByTimeAsync(25)
    expect(instances).toHaveLength(4)
    vi.useRealTimers()

    registry.stop('conn:office::work')
    expect(instances[0].closed).toBe(true)
    expect(registry.isArmed('conn:office::work')).toBe(false)
    // Tearing down one sibling must not drop the other owned scope.
    expect(instances[3].closed).toBe(false)
    expect(registry.isArmed('conn:office::less')).toBe(true)

    instances[0].emit('close', { code: 1006 })
    await new Promise(resolve => setTimeout(resolve, 50))
    expect(instances).toHaveLength(4)
    expect(registry.openUrl('conn:office::work')).toBeNull()
  })

  it('treats empty-string scope as the v1/global SSH primary but still requires baseUrl and token', async () => {
    const { createSshIsolatedKeepaliveRegistry } = await import('./ssh-isolated-keepalive')
    const { FakeWs, instances } = makeFakeWs()
    const registry = createSshIsolatedKeepaliveRegistry({ WebSocketImpl: FakeWs })

    registry.start('', { baseUrl: '', token: 'tok' })
    registry.start('', { baseUrl: 'http://127.0.0.1:9', token: '' })
    expect(instances).toHaveLength(0)
    expect(registry.isArmed('')).toBe(false)

    registry.start('', { baseUrl: 'http://127.0.0.1:53100', token: 'primary' })

    expect(instances).toHaveLength(1)
    expect(instances[0].url).toBe('ws://127.0.0.1:53100/api/ws?token=primary')
    expect(registry.isArmed('')).toBe(true)

    instances[0].emit('open')
    expect(registry.openUrl('')).toBe('ws://127.0.0.1:53100/api/ws?token=primary')

    registry.stop('')
    expect(instances[0].closed).toBe(true)
    expect(registry.isArmed('')).toBe(false)
  })
})
