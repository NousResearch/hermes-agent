import { describe, expect, it } from 'vitest'

import { connectionScopeSuffix } from './connection-scoped'

const remote = (profile: string, baseUrl = 'https://gw.example:8443') => ({
  baseUrl,
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
