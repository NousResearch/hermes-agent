import { describe, expect, it } from 'vitest'

import {
  authModeFromStatus,
  buildApiUrl,
  buildGatewayWsUrl,
  buildGatewayWsUrlWithTicket,
  connectionScopeKey,
  normalizeRemoteBaseUrl,
  pathWithProfileQuery,
} from './connection-config'

describe('normalizeRemoteBaseUrl', () => {
  it('strips trailing slash, hash, and query', () => {
    expect(normalizeRemoteBaseUrl('http://gw:9119/?x=1#h')).toBe('http://gw:9119')
  })
  it('rejects non-http(s) schemes', () => {
    expect(() => normalizeRemoteBaseUrl('ftp://gw')).toThrow()
  })
  it('rejects an empty URL', () => {
    expect(() => normalizeRemoteBaseUrl('')).toThrow()
  })
})

describe('authModeFromStatus', () => {
  it('auth_required:true → oauth (a login gate)', () => {
    expect(authModeFromStatus({ auth_required: true })).toBe('oauth')
  })
  it('everything else → token', () => {
    expect(authModeFromStatus({ auth_required: false })).toBe('token')
    expect(authModeFromStatus({})).toBe('token')
    expect(authModeFromStatus(null)).toBe('token')
  })
})

describe('ws url builders', () => {
  it('token mode embeds an (encoded) ?token=', () => {
    expect(buildGatewayWsUrl('http://gw:9119', 'tok en')).toBe('ws://gw:9119/api/ws?token=tok%20en')
  })
  it('https base → wss scheme', () => {
    expect(buildGatewayWsUrl('https://gw', 'x')).toBe('wss://gw/api/ws?token=x')
  })
  it('oauth mode embeds ?ticket=', () => {
    expect(buildGatewayWsUrlWithTicket('http://gw:9119', 'abc')).toBe('ws://gw:9119/api/ws?ticket=abc')
  })
  it('honors a base path prefix', () => {
    expect(buildGatewayWsUrl('http://gw/hermes', 't')).toBe('ws://gw/hermes/api/ws?token=t')
  })
})

describe('connectionScopeKey', () => {
  it('blank/null collapses to the primary (null)', () => {
    expect(connectionScopeKey(null)).toBeNull()
    expect(connectionScopeKey('')).toBeNull()
    expect(connectionScopeKey('  ')).toBeNull()
  })
  it('trims a named profile', () => {
    expect(connectionScopeKey(' iris ')).toBe('iris')
  })
})

describe('pathWithProfileQuery', () => {
  it('appends ?profile= for a named profile', () => {
    expect(pathWithProfileQuery('/api/model/info', 'iris')).toBe('/api/model/info?profile=iris')
  })
  it('preserves existing query params', () => {
    expect(pathWithProfileQuery('/api/model/options?force=1', 'iris')).toBe(
      '/api/model/options?force=1&profile=iris',
    )
  })
  it('does not replace an explicit profile query', () => {
    expect(pathWithProfileQuery('/api/x?profile=default', 'iris')).toBe('/api/x?profile=default')
  })
  it('no-ops for the primary (null/blank) profile', () => {
    expect(pathWithProfileQuery('/api/x', null)).toBe('/api/x')
    expect(pathWithProfileQuery('/api/x', '')).toBe('/api/x')
  })
})

describe('buildApiUrl', () => {
  it('joins base + path, honoring a prefix', () => {
    expect(buildApiUrl('http://gw:9119', '/api/status')).toBe('http://gw:9119/api/status')
    expect(buildApiUrl('http://gw/hermes', 'api/x')).toBe('http://gw/hermes/api/x')
  })
})
