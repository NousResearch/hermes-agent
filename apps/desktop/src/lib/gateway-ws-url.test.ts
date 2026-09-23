import { GatewayReauthRequiredError, isGatewayReauthRequired, resolveGatewayWsUrl } from '@hermes/shared'
import { describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'

import { resolveDesktopGatewayWsUrl } from './gateway-ws-url'

const oauthConn = { authMode: 'oauth' as const, wsUrl: 'ws://host/api/ws?ticket=stale' }
const tokenConn = { authMode: 'token' as const, wsUrl: 'ws://host/api/ws?token=abc' }

describe('desktop connection scope', () => {
  const authModes = ['token', 'oauth'] as const

  function aliasConnection(authMode: (typeof authModes)[number]) {
    return {
      authMode,
      connectionId: 'remote-device',
      profile: 'client-alias',
      wsUrl: 'wss://remote.invalid/api/ws?token=cached'
    } as HermesConnection
  }

  function registeredConnection(authMode: (typeof authModes)[number]) {
    return { ...aliasConnection(authMode), profile: 'remote-profile', registryScoped: true } as HermesConnection
  }

  function fakeDesktop(withScopedMint = true) {
    return {
      getGatewayWsUrl: vi.fn(async () => 'wss://legacy.invalid/api/ws?token=fresh'),
      ...(withScopedMint ? { getGatewayWsUrlFor: vi.fn(async () => 'wss://remote.invalid/api/ws?ticket=fresh') } : {})
    } as unknown as Window['hermesDesktop']
  }

  it.each(authModes)('an inferred connectionId keeps the legacy profile-alias mint (%s)', async authMode => {
    const desktop = fakeDesktop()

    await expect(resolveDesktopGatewayWsUrl(desktop, aliasConnection(authMode))).resolves.toContain('legacy.invalid')
    expect(desktop.getGatewayWsUrl).toHaveBeenCalledWith('client-alias')
    expect(desktop.getGatewayWsUrlFor).not.toHaveBeenCalled()
  })

  it.each(authModes)('a registry-scoped route mints against its owning connection (%s)', async authMode => {
    const desktop = fakeDesktop()

    await expect(resolveDesktopGatewayWsUrl(desktop, registeredConnection(authMode))).resolves.toContain(
      'remote.invalid'
    )
    expect(desktop.getGatewayWsUrlFor).toHaveBeenCalledWith({
      connectionId: 'remote-device',
      profile: 'remote-profile'
    })
    expect(desktop.getGatewayWsUrl).not.toHaveBeenCalled()
  })

  it('a registry-scoped flag without a connectionId still takes the legacy mint', async () => {
    const desktop = fakeDesktop()
    const scopedWithoutId = { ...registeredConnection('token'), connectionId: undefined } as HermesConnection

    await expect(resolveDesktopGatewayWsUrl(desktop, scopedWithoutId)).resolves.toContain('legacy.invalid')
    expect(desktop.getGatewayWsUrl).toHaveBeenCalledWith('remote-profile')
    expect(desktop.getGatewayWsUrlFor).not.toHaveBeenCalled()
  })

  it('a registry-scoped OAuth route without the scoped mint bridge fails instead of dialing the legacy gateway', async () => {
    const desktop = fakeDesktop(false)

    await expect(resolveDesktopGatewayWsUrl(desktop, registeredConnection('oauth'))).rejects.toThrow(
      'cannot refresh OAuth'
    )
    expect(desktop.getGatewayWsUrl).not.toHaveBeenCalled()
  })

  it('a registry-scoped token route without the scoped mint bridge reuses its cached URL', async () => {
    const desktop = fakeDesktop(false)
    const registered = registeredConnection('token')

    await expect(resolveDesktopGatewayWsUrl(desktop, registered)).resolves.toBe(registered.wsUrl)
    expect(desktop.getGatewayWsUrl).not.toHaveBeenCalled()
  })
})

describe('resolveGatewayWsUrl', () => {
  describe('oauth mode', () => {
    it('uses the freshly minted URL', async () => {
      const getGatewayWsUrl = vi.fn().mockResolvedValue('ws://host/api/ws?ticket=fresh')
      await expect(resolveGatewayWsUrl({ getGatewayWsUrl }, oauthConn)).resolves.toBe('ws://host/api/ws?ticket=fresh')
      expect(getGatewayWsUrl).toHaveBeenCalledOnce()
    })

    it('uses the structured URL returned across the Electron IPC boundary', async () => {
      const getGatewayWsUrl = vi.fn().mockResolvedValue({ ok: true, wsUrl: 'ws://host/api/ws?ticket=fresh' })

      await expect(resolveGatewayWsUrl({ getGatewayWsUrl }, oauthConn)).resolves.toBe('ws://host/api/ws?ticket=fresh')
    })

    it('throws a reauth error when the main process reports an auth rejection', async () => {
      const getGatewayWsUrl = vi.fn().mockResolvedValue({
        error: '401 cookie expired',
        needsOauthLogin: true,
        ok: false
      })

      await expect(resolveGatewayWsUrl({ getGatewayWsUrl }, oauthConn)).rejects.toBeInstanceOf(
        GatewayReauthRequiredError
      )
    })

    it('preserves the main-process auth failure as the cause', async () => {
      const getGatewayWsUrl = vi.fn().mockResolvedValue({
        error: '401 cookie expired',
        needsOauthLogin: true,
        ok: false
      })

      const error = await resolveGatewayWsUrl({ getGatewayWsUrl }, oauthConn).catch(e => e)
      expect(error).toBeInstanceOf(GatewayReauthRequiredError)
      expect((error as GatewayReauthRequiredError).cause).toMatchObject({ message: '401 cookie expired' })
    })

    it('keeps a transport failure retryable instead of demanding sign-in', async () => {
      const getGatewayWsUrl = vi.fn().mockResolvedValue({ error: 'gateway timed out', ok: false })
      const error = await resolveGatewayWsUrl({ getGatewayWsUrl }, oauthConn).catch(e => e)

      expect(error).toMatchObject({ message: 'gateway timed out' })
      expect(isGatewayReauthRequired(error)).toBe(false)
    })

    it('rethrows an unexpected transport rejection unchanged', async () => {
      const cause = new Error('socket closed')
      const getGatewayWsUrl = vi.fn().mockRejectedValue(cause)

      await expect(resolveGatewayWsUrl({ getGatewayWsUrl }, oauthConn)).rejects.toBe(cause)
    })

    it('reports a missing preload method as an app capability error, not reauth', async () => {
      const error = await resolveGatewayWsUrl({}, oauthConn).catch(e => e)

      expect(error).toMatchObject({ message: expect.stringMatching(/cannot refresh OAuth WebSocket tickets/i) })
      expect(isGatewayReauthRequired(error)).toBe(false)
    })

    it('never returns the stale cached ticket on failure', async () => {
      const getGatewayWsUrl = vi.fn().mockRejectedValue(new Error('boom'))
      const result = await resolveGatewayWsUrl({ getGatewayWsUrl }, oauthConn).catch(() => 'threw')
      expect(result).toBe('threw')
      expect(result).not.toBe(oauthConn.wsUrl)
    })
  })

  describe('token / local mode', () => {
    it('uses the minted URL when available', async () => {
      const getGatewayWsUrl = vi.fn().mockResolvedValue('ws://host/api/ws?token=fresh')
      await expect(resolveGatewayWsUrl({ getGatewayWsUrl }, tokenConn)).resolves.toBe('ws://host/api/ws?token=fresh')
    })

    it('uses a structured refreshed token URL when available', async () => {
      const getGatewayWsUrl = vi.fn().mockResolvedValue({ ok: true, wsUrl: 'ws://host/api/ws?token=fresh' })

      await expect(resolveGatewayWsUrl({ getGatewayWsUrl }, tokenConn)).resolves.toBe('ws://host/api/ws?token=fresh')
    })

    it('falls back to the cached URL when minting fails (token is long-lived)', async () => {
      const getGatewayWsUrl = vi.fn().mockRejectedValue(new Error('transient'))
      await expect(resolveGatewayWsUrl({ getGatewayWsUrl }, tokenConn)).resolves.toBe(tokenConn.wsUrl)
    })

    it('falls back to the cached URL when the preload method is absent', async () => {
      await expect(resolveGatewayWsUrl({}, tokenConn)).resolves.toBe(tokenConn.wsUrl)
    })

    it('treats a missing authMode as non-oauth (falls back safely)', async () => {
      await expect(resolveGatewayWsUrl({}, { wsUrl: tokenConn.wsUrl })).resolves.toBe(tokenConn.wsUrl)
    })
  })
})

describe('isGatewayReauthRequired', () => {
  it('detects the dedicated error class', () => {
    expect(isGatewayReauthRequired(new GatewayReauthRequiredError('x'))).toBe(true)
  })

  it('detects plain objects tagged with needsOauthLogin (from the main process)', () => {
    expect(isGatewayReauthRequired({ needsOauthLogin: true })).toBe(true)
  })

  it('rejects generic errors', () => {
    expect(isGatewayReauthRequired(new Error('connection closed'))).toBe(false)
    expect(isGatewayReauthRequired(null)).toBe(false)
    expect(isGatewayReauthRequired('string')).toBe(false)
  })
})
