/**
 * #106935: Desktop main must hold a long-lived keep-alive WebSocket for every
 * published SSH-isolated backend. Idle-exit (#101626) treats accepted WS as
 * ownership liveness; renderer sockets can drop while sshConnections still owns
 * the scope. Sticky artifacts (nonce / token file / lockfile) are NOT liveness.
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { afterEach, describe, expect, it, vi } from 'vitest'

const here = path.dirname(fileURLToPath(import.meta.url))
const mainSource = fs.readFileSync(path.join(here, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')

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

describe('main.ts wiring for SSH-isolated keep-alive WS (#106935)', () => {
  it('starts a keep-alive WebSocket after publishing sshConnections', () => {
    const publishStart = mainSource.indexOf('publish: () => {')
    expect(publishStart).toBeGreaterThan(-1)

    const publishSlice = mainSource.slice(publishStart, publishStart + 4_000)
    const setIdx = publishSlice.indexOf('sshConnections.set(scope, {')
    expect(setIdx).toBeGreaterThan(-1)

    const afterSet = publishSlice.slice(setIdx)
    expect(afterSet).toMatch(/sshIsolatedKeepalives\.start\(\s*scope/)
    expect(afterSet).toMatch(/baseUrl:\s*result\.baseUrl/)
    expect(afterSet).toMatch(/token:\s*result\.token/)
  })

  it('stops the keep-alive in teardownSshConnection so sockets cannot leak', () => {
    const fnStart = mainSource.indexOf('async function teardownSshConnection(')
    expect(fnStart).toBeGreaterThan(-1)

    const nextFn = mainSource.indexOf('\nfunction activeSshTerminalTarget(', fnStart + 1)
    const body = mainSource.slice(fnStart, nextFn === -1 ? fnStart + 1_200 : nextFn)

    expect(body).toContain('sshConnections.delete(scope)')
    expect(body).toMatch(/sshIsolatedKeepalives\.stop\(\s*scope/)
  })

  it('imports the keep-alive registry from the electron helper (not inline in main)', () => {
    expect(mainSource).toMatch(/createSshIsolatedKeepaliveRegistry/)
    expect(mainSource).toMatch(/from '\.\/ssh-isolated-keepalive'/)
  })

  it('starts keep-alive with the published scope as-is, including empty v1/global primary', () => {
    const publishStart = mainSource.indexOf('publish: () => {')
    const publishSlice = mainSource.slice(publishStart, publishStart + 4_000)
    const afterSet = publishSlice.slice(publishSlice.indexOf('sshConnections.set(scope, {'))

    // sshScopeKey(null) is '' for the settings/registry primary. Do not
    // truthiness-guard the scope before arming — that would skip the live WS.
    expect(afterSet).toMatch(/sshIsolatedKeepalives\.start\(\s*scope,\s*\{\s*baseUrl:\s*result\.baseUrl,\s*token:\s*result\.token\s*\}\)/)
    expect(afterSet).not.toMatch(/if\s*\(\s*scope\s*\)/)
    expect(afterSet).not.toMatch(/scope\s*\|\|/)
  })
})

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
