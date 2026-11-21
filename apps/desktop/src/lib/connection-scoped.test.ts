import { describe, expect, it } from 'vitest'

import { connectionScopeSuffix } from './connection-scoped'

const remote = (profile: string, baseUrl = 'https://gw.example:8443') => ({
  baseUrl,
  mode: 'remote' as const,
  profile
})

/** An SSH/URL connection: the baseUrl is the loopback tunnel it dialed through. */
const tunneled = (connectionId: string, port: number, profile = 'default') => ({
  baseUrl: `http://127.0.0.1:${port}`,
  connectionId,
  mode: 'remote' as const,
  profile
})

describe('connectionScopeSuffix', () => {
  it('is empty for a local connection', () => {
    expect(connectionScopeSuffix({ baseUrl: 'http://127.0.0.1:8000', mode: 'local', profile: 'default' })).toBe('')
  })

  it('includes the profile by default so profile-local lists stay apart', () => {
    expect(connectionScopeSuffix(remote('default'))).toBe(
      `.remote.${encodeURIComponent('https://gw.example:8443')}.default`
    )
    expect(connectionScopeSuffix(remote('k9'))).not.toBe(connectionScopeSuffix(remote('default')))
  })

  it('is stable across profile switch when includeProfile is false', () => {
    const a = connectionScopeSuffix(remote('default'), false)
    const b = connectionScopeSuffix(remote('k9'), false)

    expect(a).toBe(`.remote.${encodeURIComponent('https://gw.example:8443')}`)
    expect(a).toBe(b)
  })

  it('still isolates different gateways when includeProfile is false', () => {
    expect(connectionScopeSuffix(remote('default', 'https://gw-a.example'), false)).not.toBe(
      connectionScopeSuffix(remote('default', 'https://gw-b.example'), false)
    )
  })
})

describe('connectionScopeSuffix keys on the registry id, not the tunneled baseUrl', () => {
  it('survives the fresh loopback port every launch hands an SSH connection', () => {
    expect(connectionScopeSuffix(tunneled('mini', 57535))).toBe(`.remote.mini.default`)
    expect(connectionScopeSuffix(tunneled('mini', 52940))).toBe(connectionScopeSuffix(tunneled('mini', 57535)))
    expect(connectionScopeSuffix(tunneled('mini', 52940), false)).toBe(`.remote.mini`)
  })

  it('still keeps two registry connections apart', () => {
    expect(connectionScopeSuffix(tunneled('mini', 57535), false)).not.toBe(
      connectionScopeSuffix(tunneled('192-168-31-91', 57535), false)
    )
  })

  it('falls back to the baseUrl when the descriptor names no registry id', () => {
    expect(connectionScopeSuffix(remote('default'), false)).toBe(`.remote.${encodeURIComponent('https://gw.example:8443')}`)
  })
})
