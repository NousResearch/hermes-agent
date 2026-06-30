import { describe, expect, it } from 'vitest'

import type { DesktopConnectionConfig } from '@/global'

import { classifyGatewayStartup, summarizeGatewayDiagnostic } from './gateway-launcher-state'

function config(overrides: Partial<DesktopConnectionConfig> = {}): DesktopConnectionConfig {
  return {
    envOverride: false,
    mode: 'remote',
    profile: null,
    remoteAuthMode: 'oauth',
    remoteOauthConnected: true,
    remoteTokenPreview: null,
    remoteTokenSet: false,
    remoteUrl: 'http://mercury2:9119',
    ...overrides
  }
}

describe('classifyGatewayStartup', () => {
  it('treats missing remote URL as setup, not a configured-gateway failure', () => {
    expect(classifyGatewayStartup(config({ remoteUrl: '' }), 'Reuben requires a remote gateway URL.').kind).toBe(
      'setup'
    )
    expect(classifyGatewayStartup(null, 'Desktop IPC bridge unavailable').kind).toBe('setup')
  })

  it('treats oauth session failures as sign-in for the configured gateway', () => {
    const state = classifyGatewayStartup(
      config({ remoteOauthConnected: false }),
      'Remote Reuben gateway uses login auth, but you are not signed in.'
    )

    expect(state.kind).toBe('signin')
    expect(state.remoteUrl).toBe('http://mercury2:9119')
  })

  it('treats configured network failures as unavailable', () => {
    const state = classifyGatewayStartup(
      config(),
      'Remote Reuben gateway did not become reachable at http://mercury2:9119: connect ECONNREFUSED'
    )

    expect(state.kind).toBe('unavailable')
  })

  it('treats HTML or invalid JSON as an unexpected/incompatible gateway', () => {
    expect(classifyGatewayStartup(config(), 'Expected JSON from /api/status but got HTML').kind).toBe('unexpected')
    expect(classifyGatewayStartup(config(), 'Invalid JSON from /api/status').kind).toBe('unexpected')
  })
})

describe('summarizeGatewayDiagnostic', () => {
  it('turns common network errors into concise diagnostics', () => {
    expect(summarizeGatewayDiagnostic('getaddrinfo ENOTFOUND mercury2')).toMatch(/DNS lookup failed/)
    expect(summarizeGatewayDiagnostic('connect ECONNREFUSED 10.0.0.2:9119')).toMatch(/Connection refused/)
    expect(summarizeGatewayDiagnostic('Timed out connecting to Hermes backend after 8000ms')).toMatch(/timed out/)
  })
})
