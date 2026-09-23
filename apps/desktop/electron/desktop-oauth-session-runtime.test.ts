import { EventEmitter } from 'node:events'

import { describe, expect, it, vi } from 'vitest'

import { createDesktopOauthSessionRuntime } from './desktop-oauth-session-runtime'

function makeRuntime() {
  let ready = true
  const jars = new Map<string, any>()
  const requests: Array<{ url: string; session: any; headers: Record<string, string> }> = []
  const fromPartition = vi.fn((partition: string) => {
    let jar = jars.get(partition)

    if (!jar) {
      jar = {
        cookies: { get: vi.fn(async () => []), remove: vi.fn(async () => undefined) },
        flushStorageData: vi.fn(async () => undefined)
      }
      jars.set(partition, jar)
    }

    return jar
  })
  const request = vi.fn((options: any) => {
    const emitter = new EventEmitter() as EventEmitter & {
      setHeader: (key: string, value: string) => void
      write: (body: string) => void
      end: () => void
      abort: () => void
    }
    const headers: Record<string, string> = {}

    emitter.setHeader = (key, value) => {
      headers[key] = value
    }
    emitter.write = () => undefined
    emitter.abort = () => undefined
    emitter.end = () => {
      requests.push({ url: options.url, session: options.session, headers })
      const response = Object.assign(new EventEmitter(), { statusCode: 204, headers: {} })
      queueMicrotask(() => {
        emitter.emit('response', response)
        response.emit('end')
      })
    }

    return emitter
  })
  const registry = {
    primary: 'local',
    connections: [
      { id: 'local', kind: 'local' },
      { id: 'gateway-a', kind: 'remote', authMode: 'oauth', url: 'https://fleet.example:9119' },
      { id: 'gateway-b', kind: 'remote', authMode: 'oauth', url: 'https://fleet.example:9220' }
    ]
  }
  const runtime = createDesktopOauthSessionRuntime({
    app: { isReady: () => ready } as any,
    BrowserWindow: class {} as any,
    electronNet: { request } as any,
    session: { fromPartition } as any,
    readDesktopConnectionsRegistry: () => registry,
    readDesktopConnectionConfig: () => ({}),
    installRemoteHeaderRulesOnSession: vi.fn(),
    headersForRemoteRequest: () => ({ 'X-Fleet-Test': 'present' }),
    rememberLog: vi.fn(),
    installWindowRendererLifecycle: vi.fn(),
    finalizeGatewayDownload: vi.fn()
  })

  return { runtime, jars, fromPartition, requests, setReady: (value: boolean) => (ready = value) }
}

describe('desktop OAuth session ownership extraction', () => {
  it('keeps the portal and legacy gateway on the same ready-only cookie jar', async () => {
    const { runtime, fromPartition, setReady } = makeRuntime()

    setReady(false)
    expect(runtime.getOauthSession()).toBeNull()
    setReady(true)

    const portalJar = runtime.getOauthSession()
    expect(runtime.getOauthSessionForUrl('https://portal.nousresearch.com/api/agents')).toBe(portalJar)
    await runtime.warmOauthCookieStore()
    expect(portalJar.flushStorageData).toHaveBeenCalledTimes(1)
    expect(fromPartition).toHaveBeenCalledTimes(1)
  })

  it('keeps registered gateways in separate cookie jars and routes REST through the matching jar', async () => {
    const { runtime, requests, fromPartition } = makeRuntime()
    const a = 'https://fleet.example:9119/api/status'
    const b = 'https://fleet.example:9220/api/status'
    const jarA = runtime.getOauthSessionForUrl(a)
    const jarB = runtime.getOauthSessionForUrl(b)

    expect(jarA).not.toBe(jarB)
    expect(fromPartition).toHaveBeenCalledTimes(2)
    await expect(runtime.fetchJsonViaOauthSession(a)).resolves.toBeNull()
    await expect(runtime.fetchJsonViaOauthSession(b)).resolves.toBeNull()
    expect(requests.map(({ session }) => session)).toEqual([jarA, jarB])
    expect(requests.map(({ headers }) => headers['X-Fleet-Test'])).toEqual(['present', 'present'])
  })

  it('defers partition creation until Electron is ready and accepts a hydrated refresh cookie', async () => {
    const { runtime, jars, fromPartition, setReady } = makeRuntime()
    const url = 'https://fleet.example:9119'

    setReady(false)
    expect(runtime.getOauthSessionForUrl(url)).toBeNull()
    expect(fromPartition).not.toHaveBeenCalled()
    setReady(true)

    const jar = runtime.getOauthSessionForUrl(url)
    const partition = fromPartition.mock.calls[0][0]
    const cookieReads = [[], [], [{ name: 'hermes_session_rt', value: 'opaque' }]]
    jar.cookies.get.mockImplementation(async () => cookieReads.shift() ?? [])

    expect(await runtime.hasLiveOauthSession(url)).toBe(true)
    expect(jars.get(partition).flushStorageData).toHaveBeenCalledTimes(1)
    expect(fromPartition).toHaveBeenCalledTimes(1)
  })
})
