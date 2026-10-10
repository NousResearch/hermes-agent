import { describe, expect, it } from 'vitest'

import { hostHeaderHostname, isAcceptedHost } from '../host-validation.mjs'

describe('Desktop Web Host validation', () => {
  describe('hostHeaderHostname', () => {
    it('normalizes valid hostnames and port suffixes', () => {
      expect(hostHeaderHostname('Hermes-Desktop-Web.Elshad.Net:13043')).toBe('hermes-desktop-web.elshad.net')
      expect(hostHeaderHostname('[::1]:13043')).toBe('::1')
      expect(hostHeaderHostname('example.test')).toBe('example.test')
    })

    it('rejects malformed or URL-like Host authorities', () => {
      for (const value of [
        '',
        'example.test:not-a-port',
        'example.test/path',
        'https://example.test',
        '[::1',
        '[localhost]:13043',
        '2001:db8::1'
      ]) {
        expect(hostHeaderHostname(value), value).toBe('')
      }
    })
  })

  describe('isAcceptedHost', () => {
    it('accepts arbitrary valid hosts for an explicit all-interface bind', () => {
      expect(isAcceptedHost('hermes-desktop-web.elshad.net', '0.0.0.0')).toBe(true)
      expect(isAcceptedHost('hermes-desktop-web.elshad.net:443', '0.0.0.0')).toBe(true)
      expect(isAcceptedHost('hermes-desktop-web.elshad.net', '::')).toBe(true)
    })

    it('still rejects malformed hosts for an all-interface bind', () => {
      expect(isAcceptedHost('https://evil.test', '0.0.0.0')).toBe(false)
      expect(isAcceptedHost('evil.test:not-a-port', '0.0.0.0')).toBe(false)
      expect(isAcceptedHost('', '0.0.0.0')).toBe(false)
    })

    it('restricts loopback binds to loopback aliases', () => {
      for (const boundHost of ['127.0.0.1', 'localhost', '::1']) {
        for (const requestHost of ['127.0.0.1', 'localhost', '[::1]']) {
          expect(isAcceptedHost(requestHost, boundHost), `${boundHost} <- ${requestHost}`).toBe(true)
        }
        expect(isAcceptedHost('hermes-desktop-web.elshad.net', boundHost)).toBe(false)
      }
    })

    it('restricts explicit non-loopback binds to the bound hostname', () => {
      expect(isAcceptedHost('192.0.2.10:13043', '192.0.2.10')).toBe(true)
      expect(isAcceptedHost('other.example', '192.0.2.10')).toBe(false)
    })

    it('accepts the configured public hostname without broadening a loopback bind', () => {
      const trusted = new Set(['hermes-desktop-web.elshad.net'])
      expect(isAcceptedHost('hermes-desktop-web.elshad.net', '127.0.0.1', trusted)).toBe(true)
      expect(isAcceptedHost('untrusted.example', '127.0.0.1', trusted)).toBe(false)
    })
  })
})
