import { describe, expect, it } from 'vitest'

import { remoteAuthControlsVisible, savedCloudConnectionUrl, shouldPersistRemoteBeforeSignIn } from './gateway-settings'

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

describe('environment-controlled remote gateways', () => {
  it('keeps sign-in controls visible while URL and token fields remain environment-controlled', () => {
    expect(remoteAuthControlsVisible({ mode: 'remote', envOverride: true })).toBe(true)
  })

  it('does not overwrite saved connection settings before signing in', () => {
    expect(shouldPersistRemoteBeforeSignIn(true)).toBe(false)
    expect(shouldPersistRemoteBeforeSignIn(false)).toBe(true)
  })
})
