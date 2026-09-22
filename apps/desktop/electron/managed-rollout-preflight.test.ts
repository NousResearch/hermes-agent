import { execFile as execFileCallback } from 'node:child_process'
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { describe, expect, it } from 'vitest'

import type { RolloutPlan } from '../src/lib/managed-rollout-contract'
import {
  PROTOCOL_RESOURCE_PATH,
  TargetResolutionStore,
  ReviewTokenStore,
  acquireExactGitCache,
  canonicalPlanDigest,
  createPreflightReview,
  deleteOwnedCache,
  diffRolloutPlans,
  inspectExactGitObject,
  ownedCachePath,
  parseProtocolMetadata
} from './managed-rollout-preflight'

const execFile = promisify(execFileCallback)
const SHA = 'a'.repeat(40)
const SHA_B = 'b'.repeat(40)

function plan(head = SHA): RolloutPlan {
  return {
    target: { repositoryId: 'github.com/acme/hermes', branch: 'main', sha: SHA, protocol: 1 },
    waves: [['1'.repeat(32)], ['2'.repeat(32)]],
    concurrency: 1,
    promotionPolicy: 'manual',
    rows: [
      {
        installId: '1'.repeat(32),
        connectionId: 'conn-a',
        installationFingerprint: 'f'.repeat(64),
        sourceFingerprint: 'c'.repeat(64),
        admittedHead: head,
        requiredScopeIds: ['default'],
        eligible: true
      },
      {
        installId: '2'.repeat(32),
        connectionId: 'conn-b',
        installationFingerprint: 'e'.repeat(64),
        sourceFingerprint: 'd'.repeat(64),
        admittedHead: SHA,
        requiredScopeIds: [],
        eligible: true
      }
    ],
    retryOf: null,
    exclusions: []
  }
}

describe('managed rollout preflight', () => {
  it('strictly inspects the shared protocol-v1 metadata contract', () => {
    expect(parseProtocolMetadata(Buffer.from('{"protocol":1}\n'))).toEqual({ protocol: 1 })
    expect(() => parseProtocolMetadata(Buffer.from('{"protocol":true}'))).toThrow()
    expect(() => parseProtocolMetadata(Buffer.from('{"protocol":1.0}'))).toThrow()
    expect(() => parseProtocolMetadata(Buffer.from('{"protocol":1,"extra":0}'))).toThrow()
    expect(() => parseProtocolMetadata(Buffer.from('{"protocol":1,"protocol":1}'))).toThrow()
    expect(() => parseProtocolMetadata(Buffer.from('{"protocol":2}'))).toThrow()
    expect(() => parseProtocolMetadata(Buffer.from('/* protocol: 1 */'))).toThrow()
    expect(() => parseProtocolMetadata(new Uint8Array([0xff, 0xfe]))).toThrow()
    expect(() => parseProtocolMetadata(Buffer.alloc(4097, 0x20))).toThrow()
    expect(PROTOCOL_RESOURCE_PATH).toBe('hermes_cli/update_rollout_protocol.json')
  })

  it('reads an exact temporary Git commit object without executing target code', async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'hermes-rollout-git-'))

    try {
      await execFile('git', ['init', '--quiet'], { cwd: root })
      await execFile('git', ['config', 'user.email', 'test@example.invalid'], { cwd: root })
      await execFile('git', ['config', 'user.name', 'Hermes Test'], { cwd: root })
      await mkdir(path.join(root, 'hermes_cli'), { recursive: true })
      await writeFile(path.join(root, 'hermes_cli', 'update_rollout_protocol.json'), '{"protocol":1}\n')
      await writeFile(path.join(root, 'hermes_cli', 'target.py'), 'raise RuntimeError("must not execute")\n')
      await execFile('git', ['add', '.'], { cwd: root })
      await execFile('git', ['commit', '--quiet', '-m', 'protocol'], { cwd: root })
      const { stdout } = await execFile('git', ['rev-parse', 'HEAD'], { cwd: root })
      const sha = stdout.trim()
      const calls: string[][] = []
      const inspected = await inspectExactGitObject({
        cwd: root,
        sha,
        runGit: async args => {
          calls.push([...args])
          const result = await execFile('git', args, { cwd: root, encoding: 'buffer' })
          return { stdout: result.stdout, stderr: result.stderr }
        }
      })

      expect(inspected.protocol).toBe(1)
      expect(inspected.sha).toBe(sha)
      expect(calls).toEqual([
        ['cat-file', '-t', sha],
        ['show', `${sha}:${PROTOCOL_RESOURCE_PATH}`]
      ])
      await writeFile(path.join(root, 'hermes_cli', 'target.py'), 'raise RuntimeError("still must not execute")\n')
      await execFile('git', ['add', '.'], { cwd: root })
      await execFile('git', ['commit', '--quiet', '-m', 'advance'], { cwd: root })
      const moved = (await execFile('git', ['rev-parse', 'HEAD'], { cwd: root })).stdout.trim()
      expect(moved).not.toBe(sha)
      const retained = await inspectExactGitObject({
        cwd: root,
        sha,
        runGit: async args => {
          const result = await execFile('git', args, { cwd: root, encoding: 'buffer' })
          return { stdout: result.stdout, stderr: result.stderr }
        }
      })
      expect(retained.sha).toBe(sha)
      expect(await readFile(path.join(root, 'hermes_cli', 'target.py'), 'utf8')).toContain('must not execute')
    } finally {
      await rm(root, { recursive: true, force: true })
    }
  })

  it('deletes only an exact owned expired cache and leaves neighbors or referenced caches', async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'hermes-rollout-cache-'))

    try {
      const now = 2_000_000
      const expired = ownedCachePath(root, 'owner-a', 'cache-expired')
      const neighbor = ownedCachePath(root, 'owner-a', 'cache-expired-neighbor')
      const referenced = ownedCachePath(root, 'owner-a', 'cache-referenced')
      await Promise.all([mkdir(expired), mkdir(neighbor), mkdir(referenced)])
      const manifest = (cachePath: string, refs = 0) =>
        writeFile(
          path.join(cachePath, '.managed-rollout-cache.json'),
          JSON.stringify({
            ownerId: 'owner-a',
            cacheId: path.basename(cachePath).replace('owner-a--', ''),
            sha: SHA,
            createdAt: 1,
            lastReferencedAt: 1,
            reviewReferences: refs,
            activeReferences: 0,
            unresolvedReferences: 0
          })
        )
      await Promise.all([manifest(expired), manifest(neighbor), manifest(referenced, 1)])
      const result = await deleteOwnedCache(
        {
          path: expired,
          ownerId: 'owner-a',
          cacheId: 'cache-expired',
          sha: SHA,
          createdAt: 1,
          lastReferencedAt: 1,
          reviewReferences: 0,
          activeReferences: 0,
          unresolvedReferences: 0
        },
        { cacheRoot: root, ownerId: 'owner-a', now, ttlMs: 100 }
      )

      expect(result).toEqual({ deleted: true })
      await expect(readFile(path.join(neighbor, '.managed-rollout-cache.json'), 'utf8')).resolves.toContain('owner-a')
      await expect(readFile(path.join(referenced, '.managed-rollout-cache.json'), 'utf8')).resolves.toContain('owner-a')
      await expect(readFile(path.join(expired, '.managed-rollout-cache.json'), 'utf8')).rejects.toThrow()
    } finally {
      await rm(root, { recursive: true, force: true })
    }
  })

  it('refuses a symlink cache instead of traversing it', async () => {
    if (process.platform === 'win32') return

    const root = await mkdtemp(path.join(os.tmpdir(), 'hermes-rollout-symlink-'))
    const outside = await mkdtemp(path.join(os.tmpdir(), 'hermes-rollout-outside-'))

    try {
      const link = ownedCachePath(root, 'owner-a', 'cache-link')
      await symlink(outside, link, 'junction')
      const result = await deleteOwnedCache(
        {
          path: link,
          ownerId: 'owner-a',
          cacheId: 'cache-link',
          sha: SHA,
          createdAt: 1,
          lastReferencedAt: 1,
          reviewReferences: 0,
          activeReferences: 0,
          unresolvedReferences: 0
        },
        { cacheRoot: root, ownerId: 'owner-a', now: 2_000_000, ttlMs: 100 }
      )

      expect(result.deleted).toBe(false)
      await expect(readFile(path.join(outside, 'sentinel'), 'utf8')).rejects.toThrow()
    } finally {
      await rm(root, { recursive: true, force: true })
      await rm(outside, { recursive: true, force: true })
    }
  })

  it('revalidates an opaque token and reports only the changed row', () => {
    const store = new ReviewTokenStore({ ttlMs: 15 * 60 * 1000, tokenFactory: () => '[REDACTED]' })
    const issued = store.issue(plan(), 0)

    const renewed = store.revalidate(issued.token, plan(), 14 * 60 * 1000)
    expect(renewed.ok).toBe(true)
    if (!renewed.ok) throw new Error('expected token revalidation to succeed')
    expect(renewed.expiresAt).toBe(14 * 60 * 1000 + 15 * 60 * 1000)
    expect(renewed.planDigest).toBe(canonicalPlanDigest(plan()))

    const changed = store.revalidate(issued.token, plan(SHA_B), 14 * 60 * 1000 + 1)
    expect(changed).toMatchObject({ ok: false, code: 'plan-changed' })
    if (!changed.ok) {
      expect(changed.changes).toEqual([{ installId: '1'.repeat(32), field: 'head', before: SHA, after: SHA_B }])
    }
  })

  it('does not revalidate an expired or missing target resolution', () => {
    const resolutions = new TargetResolutionStore()
    resolutions.add({
      id: 'resolution-a',
      target: { repositoryId: 'github.com/acme/hermes', branch: 'main', sha: SHA, protocol: 1 },
      fingerprint: 'a'.repeat(64),
      cachePath: 'C:/cache/a',
      createdAt: 0,
      expiresAt: 100
    })

    expect(resolutions.get('resolution-a', 99)).not.toBeNull()
    expect(resolutions.get('resolution-a', 100)).toBeNull()
    expect(resolutions.get('missing', 0)).toBeNull()
  })

  it('bounds cache acquisition before allowing an exact target resolution', async () => {
    const calls: string[][] = []
    const result = await acquireExactGitCache({
      cacheRoot: 'C:/owned-cache',
      ownerId: 'owner-a',
      cacheId: 'cache-a',
      repository: 'https://github.com/acme/hermes.git',
      sha: SHA,
      maxBytes: 1,
      runGit: async args => {
        calls.push([...args])
        return { stdout: args[0] === 'cat-file' ? 'commit\n' : Buffer.from('{"protocol":1}') }
      },
      fileSystem: {
        ensureDirectory: async () => {},
        writeManifest: async () => {},
        inspectPath: async () => ({ exists: false, symlink: false, directory: false }),
        measureBytes: async () => 2
      }
    })

    expect(result.ok).toBe(false)
    if (result.ok !== false) throw new Error('expected cache acquisition to fail')
    expect(result.code).toBe('cache-storage-limit')
    expect(calls.some(args => args.includes('fetch'))).toBe(true)
  })

  it('refuses an existing cache without an exact ownership manifest', async () => {
    const calls: string[][] = []
    const result = await acquireExactGitCache({
      cacheRoot: 'C:/owned-cache',
      ownerId: 'owner-a',
      cacheId: 'cache-a',
      repository: 'https://github.com/acme/hermes.git',
      sha: SHA,
      runGit: async args => {
        calls.push([...args])
        return { stdout: 'commit\n' }
      },
      fileSystem: {
        ensureDirectory: async () => {},
        writeManifest: async () => {},
        inspectPath: async () => ({ exists: true, symlink: false, directory: true }),
        measureBytes: async () => 0,
        readManifest: async () => null
      }
    })

    expect(result).toMatchObject({ ok: false, code: 'cache-foreign' })
    expect(calls).toEqual([])
  })

  it('does not issue a review token for missing, expired, or incomplete admission evidence', () => {
    const reviewTokens = new ReviewTokenStore({ tokenFactory: () => '[REDACTED]' })
    const resolution = {
      id: 'resolution-a',
      target: plan().target,
      fingerprint: 'a'.repeat(64),
      cachePath: 'C:/owned-cache/owner-a--cache-a',
      createdAt: 0,
      expiresAt: 100
    }

    expect(createPreflightReview({ plan: plan(), resolution: null, reviewTokens, now: 50 }).token).toBeNull()
    expect(createPreflightReview({ plan: plan(), resolution, reviewTokens, now: 100 }).blockers).toContain(
      'target-resolution-expired'
    )
    expect(
      createPreflightReview({
        plan: { ...plan(), rows: plan().rows.map(row => ({ ...row, requiredScopeIds: null })) },
        resolution,
        reviewTokens,
        now: 50
      }).blockers
    ).toContain('scope-evidence-missing')
    const unverified = createPreflightReview({ plan: plan(), resolution, reviewTokens, now: 50 })
    expect(unverified.token).toBeNull()
    expect(unverified.blockers).toContain('reviewed-source-unverified')
  })

  it('returns deterministic exact row diffs for a changed source binding', () => {
    const before = plan()
    const after: RolloutPlan = {
      ...before,
      rows: before.rows.map((row, index) => (index === 1 ? { ...row, sourceFingerprint: 'e'.repeat(64) } : row))
    }

    expect(diffRolloutPlans(before, after)).toEqual([
      { installId: '2'.repeat(32), field: 'source', before: 'd'.repeat(64), after: 'e'.repeat(64) }
    ])

    const routeChanged: RolloutPlan = {
      ...before,
      rows: before.rows.map((row, index) => (index === 1 ? { ...row, connectionId: 'conn-c' } : row))
    }
    expect(diffRolloutPlans(before, routeChanged)).toEqual([
      {
        installId: '2'.repeat(32),
        field: 'identity',
        before: JSON.stringify({ connectionId: 'conn-b', installationFingerprint: 'e'.repeat(64) }),
        after: JSON.stringify({ connectionId: 'conn-c', installationFingerprint: 'e'.repeat(64) })
      }
    ])
  })

  it('invalidates review when the trusted inventory revision changes', () => {
    const before = { ...plan(), inventoryRevision: 'inventory-7' }
    const after = { ...before, inventoryRevision: 'inventory-8' }
    const store = new ReviewTokenStore({ tokenFactory: () => 'inventory-review' })
    const issued = store.issue(before, 0)
    const checked = store.revalidate(issued.token, after, 1)
    expect(checked.ok).toBe(false)
    if (checked.ok === false) {
      expect(checked.code).toBe('plan-changed')
      expect(checked.changes).toHaveLength(before.rows.length)
      expect(checked.changes.every(change => change.field === 'source')).toBe(true)
    }
  })
})
