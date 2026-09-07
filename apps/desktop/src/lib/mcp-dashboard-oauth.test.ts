import { afterEach, describe, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { requestGatewayForAgent } from '@/store/gateway'

import { completeMcpDesktopOAuth, McpOAuthCancelled } from './mcp-dashboard-oauth'

vi.mock('@/store/gateway', () => ({ requestGatewayForAgent: vi.fn() }))

const redirectUri = 'http://127.0.0.1:49152/callback'
const authUrl = `https://idp.example/authorize?state=expected&redirect_uri=${encodeURIComponent(redirectUri)}`
const started = { ok: true, session_id: 'flow-1', auth_url: authUrl, flow: 'pkce' }
const tools = [{ name: 'list_reports', description: 'List reports' }]

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void

  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, resolve, reject }
}

function harness() {
  const callback = deferred<{ code: string | null; state: string | null; error: string | null }>()

  const bridge = {
    listen: vi.fn().mockResolvedValue({ id: 'listener-1', redirectUri }),
    wait: vi.fn(() => callback.promise),
    cancel: vi.fn(async () => {
      callback.resolve({ code: null, state: null, error: 'cancelled' })

      return true
    })
  }

  const api = vi.fn().mockRejectedValue(new Error('Desktop OAuth must not use the remote REST callback'))

  const openExternal = vi.fn(async () => {
    callback.resolve({ code: 'code-1', state: 'expected', error: null })
  })

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { mcpOauth: bridge, api, openExternal } })
  let relayed = false
  const rpc = vi.mocked(requestGatewayForAgent)
  rpc.mockImplementation(async (_connection, _profile, method) => {
    if (method.endsWith('.start')) {
      return started
    }

    if (method.endsWith('.callback')) {
      relayed = true

      return { ok: true }
    }

    if (method.endsWith('.cancel')) {
      return { ok: true, status: 'error' }
    }

    return { ok: true, status: relayed ? 'approved' : 'pending', tools }
  })

  return { bridge, callback, api, openExternal, rpc }
}

afterEach(() => {
  vi.resetAllMocks()
  setApiRequestConnection(null)
  setApiRequestProfile(null)
})

describe('Desktop MCP client callback lifecycle', () => {
  it.each(['approved', 'cancel-start', 'cancel-poll', 'denial', 'poll', 'browser'])(
    'preserves explicit local OAuth without a bridge through %s and foreground switches',
    async outcome => {
      const { rpc, api, bridge, openExternal } = harness()
      window.hermesDesktop.mcpOauth = undefined
      setApiRequestConnection('local')
      setApiRequestProfile('origin-profile')
      let cancelled = false
      rpc.mockImplementation(async (_connection, _profile, method) => {
        if (method.endsWith('.start')) {
          setApiRequestConnection('other-gateway')
          setApiRequestProfile('other-profile')
          cancelled = outcome === 'cancel-start'

          return started
        }

        if (method.endsWith('.poll')) {
          cancelled = outcome === 'cancel-poll'

          if (outcome === 'poll') {
            throw new Error('poll failed')
          }

          return outcome === 'denial'
            ? { ok: true, status: 'error', error_message: 'access_denied' }
            : { ok: true, status: 'approved', tools }
        }

        return { ok: true }
      })

      if (outcome === 'browser') {
        openExternal.mockRejectedValue(new Error('browser failed'))
      }

      const action = completeMcpDesktopOAuth({
        serverName: 'reports',
        cancelled: () => cancelled,
        sleep: async () => {}
      })

      if (outcome === 'approved') {
        await expect(action).resolves.toMatchObject({ status: 'approved', tools })
      } else if (outcome.startsWith('cancel-')) {
        await expect(action).rejects.toBeInstanceOf(McpOAuthCancelled)
      } else {
        await expect(action).rejects.toThrow(outcome === 'denial' ? 'access_denied' : `${outcome} failed`)
      }

      expect(rpc).toHaveBeenCalledWith(
        'local',
        'origin-profile',
        'mcp.servers.oauth.start',
        { name: 'reports' },
        60_000
      )
      expect(rpc.mock.calls.every(call => call[0] === 'local' && call[1] === 'origin-profile')).toBe(true)
      expect(rpc.mock.calls.filter(call => call[2].endsWith('.start'))).toHaveLength(1)
      expect(rpc.mock.calls.filter(call => call[2].endsWith('.cancel'))).toHaveLength(outcome === 'approved' ? 0 : 1)
      expect(rpc.mock.calls.filter(call => call[2].endsWith('.callback'))).toHaveLength(0)
      expect(openExternal.mock.calls).toEqual(outcome === 'cancel-start' ? [] : [[authUrl]])
      expect(bridge.listen).not.toHaveBeenCalled()
      expect(bridge.wait).not.toHaveBeenCalled()
      expect(bridge.cancel).not.toHaveBeenCalled()
      expect(api).not.toHaveBeenCalled()
    }
  )

  it.each([
    { connectionId: null, uri: redirectUri },
    { connectionId: 'remote-gateway', uri: redirectUri },
    { connectionId: 'local', uri: 'http://remote.example:49152/callback' },
    { connectionId: 'local', uri: 'https://127.0.0.1:49152/callback' },
    { connectionId: 'local', uri: 'http://127.0.0.1:49152/not-callback' },
    { connectionId: 'local', uri: 'http://user@127.0.0.1:49152/callback' },
    { connectionId: 'local', uri: 'http://127.0.0.1:49152/callback#fragment' },
    { connectionId: 'local', uri: '' },
    { connectionId: 'local', uri: redirectUri, listenerFailure: true }
  ])('refuses unsafe compatibility paths: %j', async ({ connectionId, uri, listenerFailure }) => {
    const { rpc, api, bridge, openExternal } = harness()
    setApiRequestConnection(connectionId)

    if (listenerFailure) {
      bridge.listen.mockRejectedValue(new Error('listen failed'))
    } else {
      window.hermesDesktop.mcpOauth = undefined
    }

    rpc.mockResolvedValue({
      ...started,
      auth_url: `https://idp.example/authorize?state=expected&redirect_uri=${encodeURIComponent(uri)}`
    })
    await expect(completeMcpDesktopOAuth({ serverName: 'reports' })).rejects.toThrow()
    expect(openExternal).not.toHaveBeenCalled()
    expect(api).not.toHaveBeenCalled()

    if (connectionId !== 'local' || listenerFailure) {
      expect(rpc).not.toHaveBeenCalled()
    } else {
      expect(rpc.mock.calls.map(call => call[2])).toEqual(['mcp.servers.oauth.start', 'mcp.servers.oauth.cancel'])
    }
  })

  it('bounds pending polls even after the browser callback arrives', async () => {
    const { rpc, bridge } = harness()
    const base = rpc.getMockImplementation()!
    rpc.mockImplementation(async (...args) =>
      args[2].endsWith('.poll') ? { ok: true, status: 'pending' } : base(...args)
    )
    let now = 0
    const clock = vi.spyOn(Date, 'now').mockImplementation(() => now)

    try {
      await expect(
        completeMcpDesktopOAuth({
          serverName: 'reports',
          timeoutMs: 3000,
          sleep: async () => {
            now += 1000
          }
        })
      ).rejects.toThrow('Timed out')
      expect(bridge.cancel).toHaveBeenCalled()
    } finally {
      clock.mockRestore()
    }
  })

  it.each([null, 'local', 'remote-gateway'])(
    'relays the client callback and pins %s plus profile across foreground switches',
    async connectionId => {
      const { bridge, api, openExternal, rpc } = harness()
      setApiRequestConnection(connectionId)
      setApiRequestProfile('origin-profile')
      openExternal.mockImplementation(async () => {
        setApiRequestConnection('other-gateway')
        setApiRequestProfile('other-profile')
        const cb = { code: 'code-1', state: 'expected', error: null }
        bridge.wait.mockResolvedValue(cb)

        // The waiter was already armed before the browser opened.
        return undefined
      })
      const callbackResult = { code: 'code-1', state: 'expected', error: null }
      bridge.wait.mockResolvedValue(callbackResult)

      const result = await completeMcpDesktopOAuth({ serverName: 'reports', sleep: async () => {} })

      expect(result).toMatchObject({ status: 'approved', tools })
      expect(openExternal).toHaveBeenCalledWith(authUrl)
      expect(rpc).toHaveBeenCalledWith(
        connectionId,
        'origin-profile',
        'mcp.servers.oauth.start',
        {
          name: 'reports',
          client_redirect_uri: redirectUri
        },
        60_000
      )
      expect(rpc).toHaveBeenCalledWith(
        connectionId,
        'origin-profile',
        'mcp.servers.oauth.callback',
        {
          name: 'reports',
          session_id: 'flow-1',
          ...callbackResult
        },
        60_000
      )
      expect(rpc.mock.calls.every(call => call[0] === connectionId && call[1] === 'origin-profile')).toBe(true)
      expect(bridge.cancel).toHaveBeenCalledWith('listener-1')
      expect(api).not.toHaveBeenCalled()
    }
  )

  it.each([
    'cancel-start',
    'cancel-poll',
    'browser',
    'listen',
    'start',
    'poll',
    'relay',
    'wait',
    'legacy',
    'missing-method',
    'denial',
    'missing-bridge'
  ])('cleans up on %s without retargeting or retrying a remote callback', async failure => {
    const { bridge, callback, api, openExternal, rpc } = harness()
    setApiRequestConnection('remote-gateway')
    setApiRequestProfile('origin-profile')
    let cancelled = false
    const pendingStart = deferred<typeof started>()
    const base = rpc.getMockImplementation()!
    rpc.mockImplementation(async (connection, profile, method, params, timeout) => {
      if (method.endsWith('.start')) {
        if (failure === 'missing-method') {
          throw new Error('Unknown method: mcp.servers.oauth.start')
        }

        if (failure === 'start') {
          throw new Error('start failed')
        }

        if (failure === 'cancel-start') {
          cancelled = true
          setApiRequestConnection('other-gateway')
          setApiRequestProfile('other-profile')
          pendingStart.resolve(started)

          return pendingStart.promise
        }

        if (failure === 'legacy') {
          return {
            ...started,
            auth_url: 'https://idp.example/authorize?state=expected&redirect_uri=http%3A%2F%2Fremote%2Fcallback'
          }
        }
      }

      if (method.endsWith('.poll')) {
        if (failure === 'denial') {
          return { ok: true, status: 'error', error_message: 'access_denied' }
        }

        if (failure === 'poll') {
          throw new Error('poll failed')
        }

        if (failure === 'cancel-poll') {
          cancelled = true

          return { ok: true, status: 'pending' }
        }
      }

      if (method.endsWith('.callback') && failure === 'relay') {
        return { ok: false, error_message: 'state mismatch' }
      }

      return base(connection, profile, method, params, timeout)
    })

    if (failure === 'browser') {
      openExternal.mockRejectedValue(new Error('browser failed'))
    }

    if (failure === 'listen') {
      bridge.listen.mockRejectedValue(new Error('listen failed'))
    }

    if (failure === 'wait') {
      bridge.wait.mockRejectedValue(new Error('wait failed'))
    }

    if (failure === 'missing-bridge') {
      window.hermesDesktop.mcpOauth = undefined
    }

    if (failure === 'cancel-poll' || failure === 'poll') {
      openExternal.mockResolvedValue(undefined)
    }

    const action = completeMcpDesktopOAuth({ serverName: 'reports', cancelled: () => cancelled, sleep: async () => {} })

    if (failure.startsWith('cancel-')) {
      await expect(action).rejects.toBeInstanceOf(McpOAuthCancelled)
    } else {
      await expect(action).rejects.toThrow()
    }

    if (!['listen', 'missing-bridge'].includes(failure)) {
      expect(bridge.cancel).toHaveBeenCalledWith('listener-1')
    }

    if (!['listen', 'start', 'missing-method', 'missing-bridge'].includes(failure)) {
      expect(rpc).toHaveBeenCalledWith(
        'remote-gateway',
        'origin-profile',
        'mcp.servers.oauth.cancel',
        {
          name: 'reports',
          session_id: 'flow-1'
        },
        60_000
      )
    }

    expect(rpc.mock.calls.every(call => call[0] === 'remote-gateway' && call[1] === 'origin-profile')).toBe(true)

    if (failure === 'cancel-start' || failure === 'legacy') {
      expect(openExternal).not.toHaveBeenCalled()
    }

    expect(rpc.mock.calls.filter(call => call[2].endsWith('.start'))).toHaveLength(
      failure === 'listen' || failure === 'missing-bridge' ? 0 : 1
    )
    expect(api).not.toHaveBeenCalled()
    callback.resolve({ code: null, state: null, error: 'cancelled' })
  })
})
