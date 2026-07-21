import { describe, expect, it } from 'vitest'

import type { DesktopConnectionConfig } from '@/global'

import {
  deriveProviderShape,
  isRemoteConfig,
  isRemoteReauthError,
  isRemoteReauthFailure,
  signInLabel,
  sshFailureMessage
} from './boot-failure-reauth'

function config(overrides: Partial<DesktopConnectionConfig> = {}): DesktopConnectionConfig {
  return {
    envOverride: false,
    mode: 'remote',
    profile: null,
    remoteAuthMode: 'oauth',
    remoteOauthConnected: false,
    remoteTokenPreview: null,
    remoteTokenSet: false,
    remoteUrl: 'https://box:9119',
    cloudOrg: '',
    sshHost: '',
    sshUser: '',
    sshPort: null,
    sshKeyPath: '',
    sshRemoteHermesPath: '',
    ...overrides
  }
}

describe('isRemoteConfig', () => {
  it('true for remote/cloud with a URL, regardless of auth mode or connection', () => {
    expect(isRemoteConfig(config({ remoteAuthMode: 'token', remoteOauthConnected: false }))).toBe(true)
    expect(isRemoteConfig(config({ mode: 'cloud', remoteOauthConnected: true }))).toBe(true)
  })

  it('recognizes SSH as remote recovery without treating it as OAuth reauth', () => {
    const ssh = config({ mode: 'ssh' as never, remoteUrl: '', remoteAuthMode: 'token' }) as DesktopConnectionConfig & {
      sshHost: string
    }

    ssh.sshHost = 'remote-box'

    expect(isRemoteConfig(ssh)).toBe(true)
    expect(isRemoteReauthFailure(ssh, 'SSH authentication failed.')).toBe(false)
  })

  it('false for local, incomplete SSH, a remote with no URL, and nullish', () => {
    expect(isRemoteConfig(config({ mode: 'local' }))).toBe(false)
    expect(isRemoteConfig(config({ mode: 'ssh' as never, remoteUrl: '' }))).toBe(false)
    expect(isRemoteConfig(config({ remoteUrl: '' }))).toBe(false)
    expect(isRemoteConfig(null)).toBe(false)
  })
})

describe('isRemoteReauthFailure', () => {
  it('true for a remote, gated, disconnected gateway with a URL', () => {
    expect(isRemoteReauthFailure(config())).toBe(true)
  })

  it('false when connected and the boot error is not auth-shaped', () => {
    expect(isRemoteReauthFailure(config({ remoteOauthConnected: true }), 'Python exploded')).toBe(false)
  })

  it('true when the indicator reads connected but the boot error is auth-shaped (expired session)', () => {
    expect(
      isRemoteReauthFailure(config({ remoteOauthConnected: true }), 'Your remote gateway session has expired.')
    ).toBe(true)
  })

  it('false for a local gateway', () => {
    expect(isRemoteReauthFailure(config({ mode: 'local' }))).toBe(false)
  })

  it('true for a cloud connection with a lapsed session (cloud resolves to remote oauth)', () => {
    // A 'cloud' connection is a remote oauth backend under the hood (Q6), so a
    // lapsed cloud session is the same reauth failure as a lapsed remote one.
    expect(isRemoteReauthFailure(config({ mode: 'cloud' }))).toBe(true)
  })

  it('false for a connected cloud session', () => {
    expect(isRemoteReauthFailure(config({ mode: 'cloud', remoteOauthConnected: true }))).toBe(false)
  })

  it('false for a token (non-gated) remote gateway', () => {
    expect(isRemoteReauthFailure(config({ remoteAuthMode: 'token' }))).toBe(false)
  })

  it('false when there is no remote URL to sign in against', () => {
    expect(isRemoteReauthFailure(config({ remoteUrl: '' }))).toBe(false)
  })

  it('false for null/undefined config', () => {
    expect(isRemoteReauthFailure(null)).toBe(false)
    expect(isRemoteReauthFailure(undefined)).toBe(false)
  })
})

describe('isRemoteReauthError', () => {
  it('recognizes auth-shaped boot errors', () => {
    expect(isRemoteReauthError('Your remote gateway session has expired.')).toBe(true)
    expect(isRemoteReauthError('OAuth: please sign in')).toBe(true)
  })

  it('ignores non-auth boot errors and nullish', () => {
    expect(isRemoteReauthError('Hermes background process exited during startup.')).toBe(false)
    expect(isRemoteReauthError(null)).toBe(false)
  })
})

describe('sshFailureMessage', () => {
  it('localizes SSH failures without changing non-SSH errors', () => {
    const copy = { sshErrAuth: 'localized auth', sshErrUnknown: 'localized unknown' }
    const ssh = config({ mode: 'ssh', sshHost: 'box', remoteUrl: '' })
    expect(sshFailureMessage(ssh, 'SSH authentication failed', copy)).toBe('localized auth')
    expect(sshFailureMessage(ssh, 'unexpected failure', copy)).toBe('localized unknown')
    expect(sshFailureMessage(config(), 'raw remote error', copy)).toBe('raw remote error')
  })
})

describe('deriveProviderShape', () => {
  it('generic copy when there are no providers', () => {
    expect(deriveProviderShape([])).toEqual({ isPassword: false, providerLabel: 'your identity provider' })
    expect(deriveProviderShape(null)).toEqual({ isPassword: false, providerLabel: 'your identity provider' })
  })

  it('password shape when the sole provider supports password', () => {
    expect(
      deriveProviderShape([{ name: 'basic', displayName: 'Username & Password', supportsPassword: true }])
    ).toEqual({ isPassword: true, providerLabel: 'Username & Password' })
  })

  it('OAuth shape when the provider is a redirect IDP', () => {
    expect(deriveProviderShape([{ name: 'nous', displayName: 'Nous Research', supportsPassword: false }])).toEqual({
      isPassword: false,
      providerLabel: 'Nous Research'
    })
  })

  it('mixed deployment keeps generic OAuth copy (not every provider is password)', () => {
    const shape = deriveProviderShape([
      { name: 'basic', displayName: 'Username & Password', supportsPassword: true },
      { name: 'nous', displayName: 'Nous Research', supportsPassword: false }
    ])

    expect(shape.isPassword).toBe(false)
    expect(shape.providerLabel).toBe('Username & Password / Nous Research')
  })

  it('falls back to name when displayName is empty', () => {
    expect(deriveProviderShape([{ name: 'basic', displayName: '', supportsPassword: true }]).providerLabel).toBe(
      'basic'
    )
  })
})

describe('signInLabel', () => {
  it('password gateway gets the plain "Sign in to remote gateway" copy', () => {
    expect(signInLabel({ url: 'x', isPassword: true, providerLabel: 'Username & Password' })).toBe(
      'Sign in to remote gateway'
    )
  })

  it('OAuth gateway names the provider', () => {
    expect(signInLabel({ url: 'x', isPassword: false, providerLabel: 'Nous Research' })).toBe(
      'Sign in with Nous Research'
    )
  })

  it('null reauth falls back to the generic provider phrase', () => {
    expect(signInLabel(null)).toBe('Sign in with your identity provider')
  })
})
