import { describe, expect, it } from 'vitest'

import {
  buildLegacyRemoteBlock,
  legacyRemoteTunnelFields,
  legacyRemoteTunnelFieldsForMode
} from './legacy-remote-config'

describe('legacy Remote gateway tunnel config', () => {
  it('inherits, replaces, and explicitly clears a saved tunnel', () => {
    const url = 'http://127.0.0.1:9119'
    const saved = { host: 'alice@gw-host' }

    expect(legacyRemoteTunnelFields('remote', url, undefined, saved)).toEqual({ sshTunnel: saved })
    expect(legacyRemoteTunnelFields('remote', url, { host: 'alice@other-host' }, saved)).toEqual({
      sshTunnel: { host: 'alice@other-host' }
    })
    expect(legacyRemoteTunnelFields('remote', url, null, saved)).toEqual({})
  })

  it('builds a validated remote block and rejects a tunnel on Cloud', () => {
    expect(
      buildLegacyRemoteBlock({
        remoteUrl: 'http://127.0.0.1:9119',
        authMode: 'oauth',
        token: null,
        tokenPresent: false,
        sshTunnel: { host: 'alice@gw-host' }
      })
    ).toMatchObject({
      url: 'http://127.0.0.1:9119',
      authMode: 'oauth',
      sshTunnel: { host: 'alice@gw-host' }
    })

    expect(() => legacyRemoteTunnelFields('cloud', 'https://cloud.example.com', { host: 'alice@gw-host' })).toThrow(
      /Remote gateway/
    )
  })

  it('rejects an explicit tunnel on the Cloud save path and ignores a saved one on other modes', () => {
    const tunnel = { host: 'alice@gw-host' }

    expect(() => legacyRemoteTunnelFieldsForMode('cloud', 'https://cloud.example.com', tunnel)).toThrow(
      /Remote gateway/
    )
    expect(legacyRemoteTunnelFieldsForMode('cloud', 'https://cloud.example.com', undefined, tunnel)).toEqual({})
    expect(legacyRemoteTunnelFieldsForMode('local', '', tunnel)).toEqual({})
    expect(legacyRemoteTunnelFieldsForMode('remote', 'http://127.0.0.1:9119', undefined, tunnel)).toEqual({
      sshTunnel: tunnel
    })
  })
})
