import {
  GatewayReauthRequiredError,
  isGatewayAuthShapedError,
  isGatewayReauthRequired,
  resolveGatewayWsUrl
} from '@hermes/shared'
import { describe, expect, it, vi } from 'vitest'

const oauthConn = { authMode: 'oauth' as const, wsUrl: 'ws://host/api/ws?ticket=stale' }
const tokenConn = { authMode: 'token' as const, wsUrl: 'ws://host/api/ws?token=abc' }

describe('resolveGatewayWsUrl', () => {
  describe('oauth mode', () => {
    it('uses the freshly minted URL', async () => {
      const getGatewayWsUrl = vi.fn().mockResolvedValue('ws://host/api/ws?ticket=fresh')
      await expect(resolveGatewayWsUrl({ getGatewayWsUrl }, oauthConn)).resolves.toBe('ws://host/api/ws?ticket=fresh')
      expect(getGatewayWsUrl).toHaveBeenCalledOnce()
    })

    it('throws a reauth error instead of falling back to the stale cached ticket', async () => {
      const getGatewayWsUrl = vi.fn().mockRejectedValue(new Error('401 cookie expired'))
      await expect(resolveGatewayWsUrl({ getGatewayWsUrl }, oauthConn)).rejects.toBeInstanceOf(
        GatewayReauthRequiredError
      )
    })

    it('preserves the underlying mint failure as the cause', async () => {
      const cause = new Error('401 cookie expired')
      const getGatewayWsUrl = vi.fn().mockRejectedValue(cause)
      const error = await resolveGatewayWsUrl({ getGatewayWsUrl }, oauthConn).catch(e => e)
      expect(error).toBeInstanceOf(GatewayReauthRequiredError)
      expect((error as GatewayReauthRequiredError).cause).toBe(cause)
    })

    it('throws a reauth error when the preload cannot mint (no method)', async () => {
      await expect(resolveGatewayWsUrl({}, oauthConn)).rejects.toBeInstanceOf(GatewayReauthRequiredError)
    })

    it('never returns the stale cached ticket on failure', async () => {
      const getGatewayWsUrl = vi.fn().mockRejectedValue(new Error('boom'))
      const result = await resolveGatewayWsUrl({ getGatewayWsUrl }, oauthConn).catch(() => 'threw')
      expect(result).toBe('threw')
      expect(result).not.toBe(oauthConn.wsUrl)
    })

    it('rethrows a transport timeout as-is instead of misreporting an expired session', async () => {
      const cause = new Error('Timed out connecting to Hermes backend after 8000ms')
      const getGatewayWsUrl = vi.fn().mockRejectedValue(cause)
      const error = await resolveGatewayWsUrl({ getGatewayWsUrl }, oauthConn).catch(e => e)
      expect(error).toBe(cause)
      expect(isGatewayReauthRequired(error)).toBe(false)
    })

    it('rethrows connection-refused / unreachable-host mint failures as retryable', async () => {
      const cause = new Error('net::ERR_CONNECTION_REFUSED')
      const getGatewayWsUrl = vi.fn().mockRejectedValue(cause)
      await expect(resolveGatewayWsUrl({ getGatewayWsUrl }, oauthConn)).rejects.toBe(cause)
    })

    it('maps a 401-status mint failure to a reauth error', async () => {
      const getGatewayWsUrl = vi.fn().mockRejectedValue(new Error('401: {"detail":"session cookie expired"}'))
      await expect(resolveGatewayWsUrl({ getGatewayWsUrl }, oauthConn)).rejects.toBeInstanceOf(
        GatewayReauthRequiredError
      )
    })

    it('maps an error carrying statusCode 403 to a reauth error', async () => {
      const cause = Object.assign(new Error('Forbidden'), { statusCode: 403 })
      const getGatewayWsUrl = vi.fn().mockRejectedValue(cause)
      await expect(resolveGatewayWsUrl({ getGatewayWsUrl }, oauthConn)).rejects.toBeInstanceOf(
        GatewayReauthRequiredError
      )
    })

    it('maps an IPC-flattened main-process reauth message to a reauth error', async () => {
      const cause = new Error(
        "Error invoking remote method 'hermes:gateway:ws-url': Error: " +
          'Your remote gateway session has expired. Open Settings → Gateway and click "Sign in" again.'
      )

      const getGatewayWsUrl = vi.fn().mockRejectedValue(cause)
      await expect(resolveGatewayWsUrl({ getGatewayWsUrl }, oauthConn)).rejects.toBeInstanceOf(
        GatewayReauthRequiredError
      )
    })
  })

  describe('token / local mode', () => {
    it('uses the minted URL when available', async () => {
      const getGatewayWsUrl = vi.fn().mockResolvedValue('ws://host/api/ws?token=fresh')
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

describe('isGatewayAuthShapedError', () => {
  it('accepts explicit reauth markers, auth status codes, and canonical reauth phrasings', () => {
    expect(isGatewayAuthShapedError(new GatewayReauthRequiredError('x'))).toBe(true)
    expect(isGatewayAuthShapedError({ needsOauthLogin: true })).toBe(true)
    expect(isGatewayAuthShapedError(Object.assign(new Error('nope'), { statusCode: 401 }))).toBe(true)
    expect(isGatewayAuthShapedError(new Error('403: forbidden'))).toBe(true)
    expect(isGatewayAuthShapedError(new Error('Your remote gateway session has expired. Sign in again.'))).toBe(true)
    expect(isGatewayAuthShapedError(new Error('Remote Hermes gateway uses OAuth, but you are not signed in.'))).toBe(
      true
    )
  })

  it('rejects transport failures', () => {
    expect(isGatewayAuthShapedError(new Error('Timed out connecting to Hermes backend after 15000ms'))).toBe(false)
    expect(isGatewayAuthShapedError(new Error('net::ERR_CONNECTION_REFUSED'))).toBe(false)
    expect(isGatewayAuthShapedError(new Error('Could not connect to Hermes gateway'))).toBe(false)
    expect(isGatewayAuthShapedError(null)).toBe(false)
  })

  it('does not false-positive on incidental digit runs containing 401/403', () => {
    expect(isGatewayAuthShapedError(new Error('backend pid 84012 exited'))).toBe(false)
    expect(isGatewayAuthShapedError(new Error('http://host:64013 unreachable'))).toBe(false)
  })
})
