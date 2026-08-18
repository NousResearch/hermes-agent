import { describe, expect, it } from 'vitest'

import { coerceRemoteUrlScheme, parseSshConnectionUrl } from './remote-url'

describe('coerceRemoteUrlScheme', () => {
  it('prepends http:// to scheme-less host:port input', () => {
    expect(coerceRemoteUrlScheme('100.64.0.1:9119')).toBe('http://100.64.0.1:9119')
    expect(coerceRemoteUrlScheme('mini.tailnet-1234.ts.net:9119')).toBe('http://mini.tailnet-1234.ts.net:9119')
    expect(coerceRemoteUrlScheme('localhost:9119')).toBe('http://localhost:9119')
    expect(coerceRemoteUrlScheme('gw.example.com')).toBe('http://gw.example.com')
  })

  it('leaves explicitly schemed URLs alone', () => {
    expect(coerceRemoteUrlScheme('http://host:9119')).toBe('http://host:9119')
    expect(coerceRemoteUrlScheme('https://gw.example.com/hermes')).toBe('https://gw.example.com/hermes')
    expect(coerceRemoteUrlScheme('ws://host:9119')).toBe('ws://host:9119')
    expect(coerceRemoteUrlScheme('ftp://host:21')).toBe('ftp://host:21')
    expect(coerceRemoteUrlScheme('ssh://botnet')).toBe('ssh://botnet')
  })

  it('trims and passes through empty input', () => {
    expect(coerceRemoteUrlScheme('')).toBe('')
    expect(coerceRemoteUrlScheme('   ')).toBe('')
    expect(coerceRemoteUrlScheme('  host:9119  ')).toBe('http://host:9119')
  })
})

describe('parseSshConnectionUrl', () => {
  it('parses host, user, non-default port, and profile', () => {
    expect(parseSshConnectionUrl('ssh://botnet/')).toEqual({
      host: 'botnet',
      port: null,
      remoteProfile: '',
      user: ''
    })
    expect(parseSshConnectionUrl('ssh://alice@box:2222?profile=brawn')).toEqual({
      host: 'box',
      port: 2222,
      remoteProfile: 'brawn',
      user: 'alice'
    })
  })

  it('rejects non-ssh input', () => {
    expect(parseSshConnectionUrl('https://gateway.example.com')).toBeNull()
    expect(parseSshConnectionUrl('botnet')).toBeNull()
    expect(parseSshConnectionUrl('')).toBeNull()
  })
})

