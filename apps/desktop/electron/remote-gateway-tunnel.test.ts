import { describe, expect, it, vi } from 'vitest'

import { isExpectedTransition } from './crash-forensics'
import { remoteGatewayTunnelLoginSource, RemoteGatewayTunnels } from './remote-gateway-tunnel'

function deferred<T = void>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(r => (resolve = r))

  return { promise, resolve }
}

function fixture() {
  let nextPort = 31000
  const connections: any[] = []

  const createConnection = vi.fn(() => {
    const ssh = { open: vi.fn(async () => {}), forward: vi.fn(async () => {}), close: vi.fn(async () => {}) }
    connections.push(ssh)

    return ssh
  })

  const tunnels = new RemoteGatewayTunnels({ createConnection, allocatePort: async () => nextPort++ })

  const source = (id = 'a', port = 9119, host = id) => ({
    id,
    url: `http://127.0.0.1:${port}`,
    sshTunnel: { host }
  })

  return { tunnels, source, connections, createConnection }
}

const echo = async (url: string) => url

describe('RemoteGatewayTunnels', () => {
  it('shares one tunnel per connection and isolates hosts and gateway ports', async () => {
    const { tunnels, source, connections } = fixture()

    const [a, a2, b, custom] = await Promise.all([
      tunnels.connect(source(), 'a/default', echo),
      tunnels.connect(source(), 'a/bot', echo),
      tunnels.connect(source('b'), 'b/default', echo),
      tunnels.connect(source('custom', 9220), 'custom/default', echo)
    ])

    expect(a).toBe(a2)
    expect(new Set([a, b, custom]).size).toBe(3)
    expect(connections).toHaveLength(3)
    expect(connections[0].forward).toHaveBeenCalledExactlyOnceWith(31000, 9119, '127.0.0.1')
    expect(connections[2].forward).toHaveBeenCalledExactlyOnceWith(31002, 9220, '127.0.0.1')

    await tunnels.releaseScope('a/default')
    expect(connections[0].close).not.toHaveBeenCalled()
    await tunnels.closeOne('a')
    expect(connections[0].close).toHaveBeenCalledOnce()
    expect(connections[1].close).not.toHaveBeenCalled()
    expect(await tunnels.connect(source('b'), 'b/other', echo)).toBe(b)
    await tunnels.closeAll()
    expect(connections[1].close).toHaveBeenCalledOnce()
  })

  it('dials a direct gateway without a tunnel', async () => {
    const { tunnels, createConnection } = fixture()

    expect(await tunnels.connect({ url: 'http://gw.lan:9119' }, 'x', echo)).toBe('http://gw.lan:9119')
    expect(createConnection).not.toHaveBeenCalled()
  })

  it('retries a local bind collision on a fresh port', async () => {
    const { tunnels, source, connections, createConnection } = fixture()

    createConnection.mockImplementationOnce(() => {
      const ssh = {
        open: vi.fn(async () => {}),
        forward: vi.fn().mockRejectedValueOnce(new Error('Address already in use')).mockResolvedValue(undefined),
        close: vi.fn(async () => {})
      }

      connections.push(ssh)

      return ssh
    })

    expect(await tunnels.connect(source(), 'a', echo)).toBe('http://127.0.0.1:31001')
    expect(connections[0].forward).toHaveBeenCalledTimes(2)
    await tunnels.closeAll()
  })

  it('reopens a dead transport on the next dial', async () => {
    const { tunnels, source, connections } = fixture()
    await tunnels.connect(source(), 'a', echo)
    connections[0].isAlive = async () => false

    expect(await tunnels.connect(source(), 'a', echo)).toBe('http://127.0.0.1:31001')
    expect(connections[0].close).toHaveBeenCalledOnce()
    await tunnels.closeAll()
  })

  it('closes a tunnel whose forward or first dial fails, leaving other connections alone', async () => {
    const { tunnels, source, connections, createConnection } = fixture()

    createConnection.mockImplementationOnce(() => {
      const ssh = {
        open: vi.fn(async () => {}),
        forward: vi.fn(async () => {
          throw new Error('forward failed')
        }),
        close: vi.fn(async () => {})
      }

      connections.push(ssh)

      return ssh
    })

    await expect(tunnels.connect(source(), 'a', echo)).rejects.toThrow('forward failed')
    expect(connections[0].close).toHaveBeenCalledOnce()

    await tunnels.connect(source('b'), 'b', echo)
    await expect(
      tunnels.connect(source(), 'a', async () => {
        throw new Error('HTTP failed')
      })
    ).rejects.toThrow('HTTP failed')
    expect(connections[2].close).toHaveBeenCalledOnce()
    expect(connections[1].close).not.toHaveBeenCalled()
    await tunnels.closeAll()
  })

  it('releases one-shot holds (Test, sign-in) whether the work succeeds or fails', async () => {
    const { tunnels, source, connections } = fixture()

    expect(await tunnels.run(source(), 'test', echo)).toBe('http://127.0.0.1:31000')
    expect(connections[0].close).toHaveBeenCalledOnce()
    expect(tunnels.hasOpen()).toBe(false)

    await expect(
      tunnels.run(source(), 'login:a', async () => {
        throw new Error('window closed')
      })
    ).rejects.toThrow('window closed')
    expect(connections[1].close).toHaveBeenCalledOnce()
  })

  it('keeps a scope alive while another acquisition in that scope fails', async () => {
    const { tunnels, source, connections } = fixture()
    const kept = await tunnels.acquire(source(), 'test')

    await expect(
      tunnels.run(source(), 'test', async () => {
        throw new Error('second test failed')
      })
    ).rejects.toThrow('second test failed')
    expect(connections[0].close).not.toHaveBeenCalled()

    await kept.release()
    expect(connections[0].close).toHaveBeenCalledOnce()
  })

  it('serializes a replacement so concurrent edits never orphan a transport', async () => {
    const { tunnels, source, connections } = fixture()
    await tunnels.connect(source('a', 9119, 'old'), 'p', echo)
    const slowClose = deferred()
    connections[0].close = vi.fn(() => slowClose.promise)

    const first = tunnels.connect(source('a', 9119, 'new-1'), 'p', echo)
    const second = tunnels.connect(source('a', 9119, 'new-2'), 'p', echo)
    slowClose.resolve()

    // The later edit wins; the superseded dial fails instead of publishing a stale tunnel.
    await expect(first).rejects.toThrow(/cancelled or retired/)
    expect(await second).toBe('http://127.0.0.1:31002')
    expect(connections).toHaveLength(3)
    expect(connections[1].close).toHaveBeenCalledOnce()
    await tunnels.closeAll()
    expect(connections[2].close).toHaveBeenCalledOnce()
    expect(tunnels.hasOpen()).toBe(false)
  })

  it('quit waits for transports still closing and refuses new tunnels', async () => {
    const { tunnels, source, connections } = fixture()
    const lease = await tunnels.acquire(source(), 'p')
    const slowClose = deferred()
    connections[0].close = vi.fn(() => slowClose.promise)

    const releasing = lease.release()
    expect(tunnels.hasOpen()).toBe(true)

    let quitDone = false
    const quitting = tunnels.closeAll().then(() => (quitDone = true))
    await Promise.resolve()
    expect(quitDone).toBe(false)

    slowClose.resolve()
    await Promise.all([releasing, quitting])
    expect(tunnels.hasOpen()).toBe(false)
    const refused = await tunnels.connect(source(), 'p', echo).catch((error: unknown) => error)
    expect(isExpectedTransition(refused)).toBe(true)
  })

  it('never publishes a dial that completes after its connection was removed', async () => {
    const { tunnels, source, connections, createConnection } = fixture()
    const opened = deferred()

    createConnection.mockImplementationOnce(() => {
      const ssh = {
        open: vi.fn(() => opened.promise),
        forward: vi.fn(async () => {}),
        close: vi.fn(async () => {})
      }

      connections.push(ssh)

      return ssh
    })

    const pending = tunnels.connect(source(), 'a', echo)
    const rejected = expect(pending).rejects.toThrow(/cancel|retired/i)
    await vi.waitFor(() => expect(connections[0].open).toHaveBeenCalled())
    const closing = tunnels.closeOne('a')
    opened.resolve()
    await closing
    await rejected
    expect(connections[0].forward).not.toHaveBeenCalled()
    expect(connections[0].close).toHaveBeenCalledOnce()
  })
})

describe('signing in through a tunnel', () => {
  it('maps the dynamic local URL back to its connection, and forgets it once closed', async () => {
    const { tunnels, source } = fixture()
    const lease = await tunnels.acquire(source(), 'login:a')

    expect(lease.url).toBe('http://127.0.0.1:31000')
    expect(tunnels.connectionIdForUrl(`${lease.url}/api/status`)).toBe('a')
    expect(tunnels.connectionIdForUrl('http://127.0.0.1:9119')).toBe('')

    await lease.release()
    expect(tunnels.connectionIdForUrl(lease.url)).toBe('')
  })

  it('takes the tunnel from the draft or the saved entry by id, never from a URL alone', () => {
    const url = 'http://127.0.0.1:9119'
    const saved = { id: 'gw-host', url, sshTunnel: { host: 'gw-host' } }
    const other = { id: 'other', url, sshTunnel: { host: 'other-host' } }
    const registry = { connections: [saved, other, { id: 'plain', url: 'http://127.0.0.1:9220' }] }

    expect(remoteGatewayTunnelLoginSource(registry, { sshTunnel: { host: 'box' } }, 'draft', url)).toEqual({
      id: 'draft',
      url,
      sshTunnel: { host: 'box' }
    })
    expect(remoteGatewayTunnelLoginSource(registry, { sshTunnel: null }, 'gw-host', url).sshTunnel).toBeUndefined()
    expect(remoteGatewayTunnelLoginSource(registry, {}, 'gw-host', url)).toEqual(saved)
    expect(remoteGatewayTunnelLoginSource(registry, undefined, 'other', url)).toEqual(other)
    // Unknown id + a URL several tunnels share: no guess, the URL is dialed as given.
    expect(remoteGatewayTunnelLoginSource(registry, undefined, 'unknown', url)).toEqual({ url })
    expect(remoteGatewayTunnelLoginSource(registry, undefined, '', url)).toEqual({ url })
  })
})
