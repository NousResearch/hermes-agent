import { describe, expect, it } from 'vitest'

import { MCP_DEEPLINK_MAX_CONFIG_BYTES, parseMcpInstallDeepLink } from './mcp-deeplink'

const b64url = (value: unknown) =>
  btoa(JSON.stringify(value)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')

describe('parseMcpInstallDeepLink', () => {
  it('accepts a url-shaped config (base64url)', () => {
    const result = parseMcpInstallDeepLink({
      name: 'context7',
      config: b64url({ url: 'https://mcp.context7.com/mcp', headers: { Authorization: 'Bearer x' } })
    })

    expect(result).toEqual({
      ok: true,
      request: {
        name: 'context7',
        transport: 'http',
        config: { url: 'https://mcp.context7.com/mcp', headers: { Authorization: 'Bearer x' } }
      }
    })
  })

  it('accepts a command-shaped config (standard base64 with padding)', () => {
    const config = { command: 'npx', args: ['-y', '@modelcontextprotocol/server-filesystem', '/tmp'] }
    const result = parseMcpInstallDeepLink({ name: 'fs.local-1', config: btoa(JSON.stringify(config)) })

    expect(result).toEqual({ ok: true, request: { name: 'fs.local-1', transport: 'stdio', config } })
  })

  it('rejects names outside ^[A-Za-z0-9._-]{1,64}$', () => {
    const config = b64url({ url: 'https://example.com/mcp' })

    for (const name of ['', 'has space', 'näme', 'a/b', 'a'.repeat(65)]) {
      expect(parseMcpInstallDeepLink({ name, config })).toEqual({ ok: false, error: 'invalid_name' })
    }
  })

  it('rejects a missing or undecodable config param', () => {
    expect(parseMcpInstallDeepLink({ name: 'x' })).toEqual({ ok: false, error: 'missing_config' })
    expect(parseMcpInstallDeepLink({ name: 'x', config: '!!!not-base64!!!' })).toEqual({
      ok: false,
      error: 'bad_encoding'
    })
    // Valid base64, not JSON.
    expect(parseMcpInstallDeepLink({ name: 'x', config: btoa('not json') })).toEqual({
      ok: false,
      error: 'bad_encoding'
    })
  })

  it('rejects non-object configs', () => {
    for (const value of ['a string', 42, null, true, ['array']]) {
      expect(parseMcpInstallDeepLink({ name: 'x', config: b64url(value) })).toEqual({
        ok: false,
        error: 'not_an_object'
      })
    }
  })

  it('rejects configs with neither url nor command', () => {
    expect(parseMcpInstallDeepLink({ name: 'x', config: b64url({ env: { A: '1' } }) })).toEqual({
      ok: false,
      error: 'invalid_shape'
    })
    // Non-string command.
    expect(parseMcpInstallDeepLink({ name: 'x', config: b64url({ command: ['npx'] }) })).toEqual({
      ok: false,
      error: 'invalid_shape'
    })
  })

  it('rejects non-http(s) url schemes', () => {
    for (const url of ['javascript:alert(1)', 'file:///etc/passwd', 'ftp://x', 'not a url']) {
      expect(parseMcpInstallDeepLink({ name: 'x', config: b64url({ url }) })).toEqual({
        ok: false,
        error: 'invalid_url'
      })
    }
  })

  it('rejects ambiguous configs carrying both url and command', () => {
    expect(
      parseMcpInstallDeepLink({ name: 'x', config: b64url({ url: 'https://example.com', command: 'rm' }) })
    ).toEqual({ ok: false, error: 'invalid_shape' })
  })

  it('rejects oversized payloads', () => {
    const big = { url: 'https://example.com/mcp', padding: 'x'.repeat(MCP_DEEPLINK_MAX_CONFIG_BYTES) }

    expect(parseMcpInstallDeepLink({ name: 'x', config: b64url(big) })).toEqual({
      ok: false,
      error: 'payload_too_large'
    })
  })
})
