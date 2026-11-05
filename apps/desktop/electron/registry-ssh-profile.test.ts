import { describe, expect, it } from 'vitest'

import { pathForRegistryBackendRequest } from './connection-config'
import { registrySshRemoteProfile } from './registry-ssh-profile'
import { managedSshConfig } from './ssh-shared-profile-backend'

describe('registered SSH profile ownership', () => {
  it('preserves the saved launch profile only for the default connection door', () => {
    expect(registrySshRemoteProfile('aria', undefined)).toBe('aria')
    expect(registrySshRemoteProfile('aria', 'default')).toBe('aria')
    expect(registrySshRemoteProfile(undefined, 'default')).toBe('')
  })

  it('dials the explicitly selected remote profile and keeps its session filter', () => {
    const remoteProfile = registrySshRemoteProfile('aria', 'turing')
    expect(remoteProfile).toBe('turing')
    const path = '/api/profiles/sessions?profile=turing&limit=40'
    expect(pathForRegistryBackendRequest(path, 'turing', { mode: 'remote', remoteProfile })).toBe(path)
  })

  it('uses the same helper on the production managed SSH config path', () => {
    const config = managedSshConfig(
      {
        id: 'macbook',
        kind: 'ssh',
        label: 'MacBook',
        host: 'macbook.local',
        user: 'guto',
        port: 22,
        keyPath: '~/.ssh/id_ed25519',
        remoteHermesPath: 'hermes',
        remoteProfile: 'aria'
      },
      'turing'
    )

    expect(config.remoteProfile).toBe('turing')
  })
})
