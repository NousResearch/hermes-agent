import { describe, expect, it } from 'vitest'

import { normalizeGatewaySettingsState } from './gateway-settings'

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

