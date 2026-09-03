import { describe, expect, it } from 'vitest'

import { normalizeGatewaySettingsState, savedCloudConnectionUrl } from './gateway-settings'

describe('normalizeGatewaySettingsState', () => {
  it('fills missing and undefined persisted fields with canonical defaults', () => {
    const normalized = normalizeGatewaySettingsState({
      mode: 'remote',
      remoteAuthMode: undefined,
      remoteUrl: 'https://gateway.example'
    })

    expect(normalized.mode).toBe('remote')
    expect(normalized.remoteAuthMode).toBe('token')
    expect(normalized.remoteUrl).toBe('https://gateway.example')
    expect(normalized.sshHost).toBe('')
    expect(normalized.sshPort).toBeNull()
    expect(normalized.secureTokenStorage).toBe(true)
  })

  it('returns an independent default state for invalid persisted data', () => {
    const first = normalizeGatewaySettingsState(null)
    const second = normalizeGatewaySettingsState(undefined)

    expect(first).toEqual(second)
    expect(first).not.toBe(second)
  })
})

describe('savedCloudConnectionUrl', () => {
  it('normalizes the URL of a persisted cloud connection', () => {
    expect(savedCloudConnectionUrl({ mode: 'cloud', remoteUrl: ' HTTPS://AGENT.EXAMPLE/ ' })).toBe(
      'https://agent.example'
    )
  })

  it('does not treat a stale cloud URL on a local config as connected', () => {
    expect(savedCloudConnectionUrl({ mode: 'local', remoteUrl: 'https://agent.example' })).toBe('')
  })

  it('does not treat a remote gateway URL as a connected cloud agent', () => {
    expect(savedCloudConnectionUrl({ mode: 'remote', remoteUrl: 'https://agent.example' })).toBe('')
  })
})
