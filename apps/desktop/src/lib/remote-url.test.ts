import { describe, expect, it } from 'vitest'

import { coerceRemoteUrlScheme } from './remote-url'

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
  })

  it('trims and passes through empty input', () => {
    expect(coerceRemoteUrlScheme('')).toBe('')
    expect(coerceRemoteUrlScheme('   ')).toBe('')
    expect(coerceRemoteUrlScheme('  host:9119  ')).toBe('http://host:9119')
  })
})
