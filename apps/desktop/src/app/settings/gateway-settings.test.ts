import { describe, expect, it } from 'vitest'

import { savedCloudConnectionUrl, stateForRemoteMode } from './gateway-settings'

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

describe('stateForRemoteMode', () => {
  const state = {
    cloudOrg: 'cloud-org',
    connections: [
      {
        id: 'home-lab',
        name: 'Home Lab',
        remoteAuthMode: 'oauth' as const,
        remoteOauthConnected: true,
        remoteTokenPreview: null,
        remoteTokenSet: false,
        remoteUrl: 'https://home.example'
      }
    ],
    envOverride: false,
    mode: 'cloud' as const,
    remoteAuthMode: 'token' as const,
    remoteOauthConnected: false,
    remoteTokenPreview: 'cl…ud',
    remoteTokenSet: true,
    remoteUrl: 'https://cloud.example',
    selectedConnectionId: 'home-lab',
    selectedConnectionName: 'Home Lab'
  }

  it('restores the selected saved remote when leaving Cloud globally', () => {
    expect(stateForRemoteMode(state, null)).toMatchObject({
      mode: 'remote',
      remoteAuthMode: 'oauth',
      remoteOauthConnected: true,
      remoteTokenSet: false,
      remoteUrl: 'https://home.example'
    })
  })

  it('does not leak the global saved remote into a profile override editor', () => {
    expect(stateForRemoteMode(state, 'work')).toMatchObject({
      mode: 'remote',
      remoteAuthMode: 'token',
      remoteUrl: 'https://cloud.example'
    })
  })
})
