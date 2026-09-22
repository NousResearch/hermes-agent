import { execFile as execFileCallback } from 'node:child_process'
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { describe, expect, it } from 'vitest'

import type { ReviewedSourceBinding } from '../src/lib/managed-rollout-contract'
import { createPreflightReview, ReviewTokenStore } from './managed-rollout-preflight'
import {
  isVerifiedAssurance,
  isVerifiedGitSource,
  verifyApplicableAssurance,
  verifyReviewedGitSource
} from './managed-rollout-assurance'

const execFile = promisify(execFileCallback)
const ORIGIN = 'https://github.com/acme/hermes.git'
const REPOSITORY = 'github.com/acme/hermes'
const FINGERPRINT = 'b'.repeat(64)
const RECEIPT = 'c'.repeat(64)

function envelope(sha: string, overrides: Record<string, unknown> = {}): Uint8Array {
  return Buffer.from(JSON.stringify({
    schema: 1,
    profile: 'managed-ssh-v1',
    generation: 4,
    repositoryId: REPOSITORY,
    targetSha: sha,
    sourceFingerprint: FINGERPRINT,
    observedAt: '2026-09-21T00:00:00.000Z',
    expiresAt: '2026-09-23T00:00:00.000Z',
    controls: [{ id: 'source-integrity', required: true, result: 'pass', receiptSha256: RECEIPT }],
    ...overrides
  }))
}

function assuranceReader(
  raw: Uint8Array | null,
  requiredControlIds: readonly string[] | null = ['source-integrity'],
  generation = 4
) {
  return {
    readEvidence: async () => raw,
    readProfile: async () => requiredControlIds === null ? null : { generation, requiredControlIds }
  }
}

describe('managed rollout source and assurance admission', () => {
  it('checks exact Git object, origin, root, ref and protocol on a disposable repository', async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'hermes-assurance-'))
    const git = async (...args: string[]) => (await execFile('git', args, { cwd: root })).stdout.trim()
    try {
      await git('init', '--quiet')
      await git('config', 'user.email', 'test@example.invalid')
      await git('config', 'user.name', 'Hermes Test')
      await git('remote', 'add', 'origin', ORIGIN)
      await mkdir(path.join(root, 'hermes_cli'))
      await writeFile(path.join(root, 'hermes_cli', 'update_rollout_protocol.json'), '{"protocol":1}\n')
      await git('add', '.')
      await git('commit', '--quiet', '-m', 'reviewed B')
      const reviewed = await git('rev-parse', 'HEAD')
      await git('branch', '-M', 'main')
      await writeFile(path.join(root, 'later.txt'), 'tip C\n')
      await git('add', '.')
      await git('commit', '--quiet', '-m', 'tip C')
      const tip = await git('rev-parse', 'HEAD')
      await git('update-ref', 'refs/remotes/origin/main', tip)
      const now = Date.parse('2026-09-22T00:00:00.000Z')
      const assurance = await verifyApplicableAssurance({
        profile: 'managed-ssh-v1', repositoryId: REPOSITORY, targetSha: reviewed,
        sourceFingerprint: FINGERPRINT, generation: 4, now
      }, assuranceReader(envelope(reviewed)))
      const source: ReviewedSourceBinding = {
        repositoryRoot: root,
        originUrl: ORIGIN,
        resolvedRef: 'refs/remotes/origin/main',
        targetSha: reviewed,
        assuranceProfile: 'managed-ssh-v1',
        assuranceEvidenceSha256: assurance.evidenceSha256,
        assuranceGeneration: 4
      }
      const expected = {
        target: { repositoryId: REPOSITORY, branch: 'main', sha: reviewed, protocol: 1 as const },
        trustedOriginUrl: ORIGIN,
        repositoryRoot: root,
        branch: 'main'
      }
      const reader = { git: async (args: readonly string[], cwd: string) =>
        (await execFile('git', [...args], { cwd, encoding: 'buffer', env: { ...process.env, GIT_NO_LAZY_FETCH: '1', GIT_TERMINAL_PROMPT: '0' } })).stdout }

      expect(isVerifiedGitSource(source)).toBe(false)
      const verified = await verifyReviewedGitSource(source, expected, reader)
      expect(isVerifiedGitSource(verified)).toBe(true)
      expect(verified.targetSha).toBe(reviewed)
      expect(verified.targetSha).not.toBe(tip)
      const installId = '1'.repeat(32)
      const plan = {
        target: expected.target, waves: [[installId]], concurrency: 1,
        promotionPolicy: 'manual' as const,
        rows: [{
          installId, connectionId: 'ssh-a', installationFingerprint: 'a'.repeat(64),
          sourceFingerprint: FINGERPRINT, admittedHead: reviewed, requiredScopeIds: [],
          eligible: true, reviewedSource: verified
        }],
        retryOf: null, exclusions: []
      }
      const review = createPreflightReview({
        plan,
        resolution: {
          id: 'resolution-a', target: expected.target, fingerprint: 'a'.repeat(64),
          cachePath: path.join(root, 'cache'), createdAt: now - 1_000, expiresAt: now + 60_000
        },
        reviewTokens: new ReviewTokenStore({ tokenFactory: () => 'review-token' }), now,
        verifiedSources: new Map([[installId, verified]]),
        verifiedAssurance: new Map([[installId, assurance]])
      })
      expect(review.blockers).toEqual([])
      expect(review.token).toBe('review-token')
      expect(createPreflightReview({
        plan: { ...plan, rows: [{ ...plan.rows[0], sourceFingerprint: 'd'.repeat(64) }] },
        resolution: {
          id: 'resolution-a', target: expected.target, fingerprint: 'a'.repeat(64),
          cachePath: path.join(root, 'cache'), createdAt: now - 1_000, expiresAt: now + 60_000
        },
        reviewTokens: new ReviewTokenStore(), now,
        verifiedSources: new Map([[installId, verified]]),
        verifiedAssurance: new Map([[installId, assurance]])
      }).blockers).toContain('assurance-evidence-stale-or-mismatched')
      await expect(verifyReviewedGitSource(source, { ...expected, trustedOriginUrl: 'https://github.com/other/repo.git' }, reader))
        .rejects.toThrow('reviewed-origin-mismatch')
      await git('remote', 'set-url', 'origin', 'https://github.com/other/repo.git')
      await expect(verifyReviewedGitSource(source, expected, reader)).rejects.toThrow('reviewed-origin-mismatch')
    } finally {
      await rm(root, { recursive: true, force: true })
    }
  })

  it('requires an independently read, current, passing exact-object assurance record', async () => {
    const sha = 'a'.repeat(40)
    const expected = {
      profile: 'managed-ssh-v1', repositoryId: REPOSITORY, targetSha: sha,
      sourceFingerprint: FINGERPRINT, generation: 4, now: Date.parse('2026-09-22T00:00:00.000Z')
    }
    await expect(verifyApplicableAssurance(expected, assuranceReader(null)))
      .rejects.toThrow('assurance-evidence-missing')
    await expect(verifyApplicableAssurance(expected, assuranceReader(envelope(sha), null)))
      .rejects.toThrow('assurance-profile-unavailable')
    await expect(verifyApplicableAssurance(expected, assuranceReader(envelope(sha), ['source-integrity'], 5)))
      .rejects.toThrow('assurance-profile-unavailable')
    const verified = await verifyApplicableAssurance(expected, assuranceReader(envelope(sha)))
    expect(isVerifiedAssurance(verified)).toBe(true)
    expect(verified.evidenceSha256).toMatch(/^[0-9a-f]{64}$/)
    await expect(verifyApplicableAssurance(expected, assuranceReader(envelope(sha, { generation: 3 }))))
      .rejects.toThrow('assurance-evidence-stale-or-mismatched')
    await expect(verifyApplicableAssurance(expected, assuranceReader(envelope(sha, { targetSha: 'd'.repeat(40) }))))
      .rejects.toThrow('assurance-evidence-stale-or-mismatched')
    await expect(verifyApplicableAssurance(expected, assuranceReader(envelope(sha, {
      controls: [{ id: 'source-integrity', required: true, result: 'fail', receiptSha256: RECEIPT }]
    })))).rejects.toThrow('assurance-required-control-not-passed')
    await expect(verifyApplicableAssurance(expected, assuranceReader(envelope(sha), ['missing-required-control'])))
      .rejects.toThrow('assurance-required-control-not-passed')
    await expect(verifyApplicableAssurance(
      { ...expected, now: Date.parse('2026-09-24T00:00:00.000Z') }, assuranceReader(envelope(sha))
    )).rejects.toThrow('assurance-evidence-stale-or-mismatched')
  })
})
