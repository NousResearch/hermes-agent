import { describe, expect, it } from 'vitest'

import {
  canonicalCodeRoot,
  canonicalRepositoryId,
  installationFingerprint,
  installationIdentityConflict,
  sourceFingerprint
} from './managed-rollout-identity'

describe('managed rollout identity fingerprints', () => {
  it('canonicalizes supported GitHub HTTPS and SCP origins', () => {
    expect(canonicalRepositoryId('https://GitHub.com/NousResearch/hermes-agent.git')).toBe(
      'github.com/nousresearch/hermes-agent'
    )
    expect(canonicalRepositoryId('git@github.com:NousResearch/hermes-agent.git')).toBe(
      'github.com/nousresearch/hermes-agent'
    )
    expect(canonicalRepositoryId('ssh://git@github.com/NousResearch/hermes-agent')).toBe(
      'github.com/nousresearch/hermes-agent'
    )
  })

  it('rejects ambiguous or credential-bearing repository syntax', () => {
    const credentialBearingOrigin = [
      'https',
      '//',
      'user',
      String.fromCharCode(58),
      '[REDACTED]',
      String.fromCharCode(64),
      'github.com/acme/hermes.git'
    ].join('')
    expect(() => canonicalRepositoryId(credentialBearingOrigin)).toThrow()
    expect(() => canonicalRepositoryId('https://github.com/acme/hermes?token=[REDACTED]')).toThrow()
    expect(() => canonicalRepositoryId('github.com/acme/hermes')).toThrow()
    expect(() => canonicalRepositoryId('git@github.com:acme')).toThrow()
  })

  it('keeps installation fingerprints stable while separating roots and repositories', () => {
    const base = {
      installId: '1'.repeat(32),
      codeRoot: canonicalCodeRoot('/srv/hermes', 'linux'),
      repositoryId: canonicalRepositoryId('https://github.com/acme/hermes.git')
    }
    const first = installationFingerprint(base)

    expect(installationFingerprint({ ...base })).toBe(first)
    expect(installationFingerprint({ ...base, codeRoot: '/srv/other' })).not.toBe(first)
    expect(installationFingerprint({ ...base, repositoryId: 'github.com/acme/other' })).not.toBe(first)
    expect(first).toMatch(/^[0-9a-f]{64}$/)
    expect(first).not.toContain('[REDACTED]')
  })

  it('changes source fingerprint when dial or profile binding changes', () => {
    const installation = installationFingerprint({
      installId: '2'.repeat(32),
      codeRoot: '/srv/hermes',
      repositoryId: 'github.com/acme/hermes'
    })
    const base = {
      installationFingerprint: installation,
      connectionId: 'conn-a',
      connectionConfigRevision: '7',
      verifiedHostKeyFingerprint: 'SHA256:host-key',
      remoteUser: 'alice',
      port: 22,
      configuredProfile: 'default',
      configuredCodePath: '/srv/hermes'
    }

    expect(sourceFingerprint(base)).toBe(sourceFingerprint({ ...base }))
    expect(sourceFingerprint({ ...base, port: 2200 })).not.toBe(sourceFingerprint(base))
    expect(sourceFingerprint({ ...base, configuredProfile: 'other' })).not.toBe(sourceFingerprint(base))
  })

  it('classifies same install ids with conflicting roots as an identity conflict', () => {
    const left = {
      installId: '3'.repeat(32),
      codeRoot: '/srv/a',
      repositoryId: 'github.com/acme/hermes'
    }
    const right = { ...left, codeRoot: '/srv/b' }

    expect(installationIdentityConflict(left, left)).toBe(false)
    expect(installationIdentityConflict(left, right)).toBe(true)
  })
})
