/**
 * Tests for electron/profile-migration.ts — pure migration-decision helpers for
 * the active-profile.json first-boot seeding.
 *
 * Run with: `vitest run` (wired via the `electronNative` project in
 * `apps/desktop/vitest.config.ts`, which discovers tests under `electron/`).
 *
 * These tests cover the four axes the maintainer explicitly required:
 *   1. precedence — legacy > single-running-gateway > state.db heuristic
 *   2. stale PID rejection — recycled PID must not pass as a running gateway
 *   3. fallback behavior — single-profile installs and missing files are no-ops
 *   4. remote-boot path — verified by code review (the migration is moved to the
 *      top of startHermes() in main.ts, before primaryProfileKey() is read); the
 *      pure decision logic that the function relies on is covered below.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import type { Dirent } from 'node:fs'

import {
  decideMigration,
  findRunningGatewayProfiles,
  listProfileDirs,
  migrateActiveProfileIfMissing,
  PROFILE_SCORE_MIN_SIZE_BYTES,
  readLegacyActiveProfile,
  scoreStateDb
} from './profile-migration'

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

// Mirrors the production PROFILE_NAME_RE in main.ts (regex matches "default";
// callers like readLegacyActiveProfile must reject "default" explicitly).
const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/

// Production-shaped validator: regex matches, but 'default' is excluded because
// it's the implicit fallback, never a user-chosen CLI value.
const isValidProfileName = (n: string) => n !== 'default' && PROFILE_NAME_RE.test(n)

const NOW = 1_700_000_000_000

type FileFixture = { content?: string; size?: number; mtime?: number; dir?: boolean }

function makeFs(files: Record<string, FileFixture>) {
  const map = new Map(Object.entries(files))

  const existsSync = (p: string) => map.has(p)

  const readFileSync = (p: string, _enc: 'utf8') => {
    const e = map.get(p)
    if (!e || e.content === undefined) throw new Error(`ENOENT: ${p}`)
    return e.content
  }

  const statSync = (p: string) => {
    const e = map.get(p)
    if (!e || e.size === undefined) throw new Error(`ENOENT: ${p}`)
    return { size: e.size, mtimeMs: e.mtime ?? NOW - 86_400_000 }
  }

  const readdirSync = (p: string, _options?: { withFileTypes?: boolean }): Dirent[] => {
    const names = new Set<string>()
    // 1. Explicit dir markers at this exact level.
    for (const [fullPath, fixture] of map.entries()) {
      if (fullPath === p && fixture.dir) {
        // This is the dir itself; not a child. Skip.
        continue
      }
      if (fullPath.startsWith(p + '/')) {
        const rest = fullPath.slice(p.length + 1)
        if (!rest.includes('/') && fixture.dir) {
          names.add(rest)
        }
      }
    }
    // 2. Implicit directories: any nested path under `p` implies the intermediate
    // directory exists (mirrors how mkdir({recursive:true}) populates the tree).
    for (const key of map.keys()) {
      if (!key.startsWith(p + '/')) continue
      const rest = key.slice(p.length + 1)
      const parts = rest.split('/')
      if (parts.length < 2) continue
      names.add(parts[0])
    }
    const entries: Dirent[] = []
    for (const name of names) {
      entries.push({
        name,
        isDirectory: () => true,
        isFile: () => false,
        isBlockDevice: () => false,
        isCharacterDevice: () => false,
        isSymbolicLink: () => false,
        isFIFO: () => false,
        isSocket: () => false
      } as unknown as Dirent)
    }
    if (entries.length === 0 && !map.has(p)) throw new Error(`ENOENT: ${p}`)
    return entries
  }

  return { existsSync, readFileSync, statSync, readdirSync }
}

function baseDeps(overrides: Record<string, unknown> = {}) {
  const fs = makeFs({})
  return {
    legacyActivePath: '/home/u/.hermes/active_profile',
    profilesRoot: '/home/u/.hermes/profiles',
    existsSync: fs.existsSync,
    readFileSync: fs.readFileSync,
    statSync: fs.statSync,
    readdirSync: fs.readdirSync,
    isHermesProcess: () => true,
    now: () => NOW,
    writeJson: () => {},
    isValidProfileName,
    ...overrides
  }
}

// ---------------------------------------------------------------------------
// readLegacyActiveProfile
// ---------------------------------------------------------------------------

test('readLegacyActiveProfile returns null when file missing', () => {
  assert.equal(
    readLegacyActiveProfile('/missing', () => { throw new Error('ENOENT') }, isValidProfileName),
    null
  )
})

test('readLegacyActiveProfile returns trimmed name on success', () => {
  assert.equal(
    readLegacyActiveProfile('/p', () => '  coder  \n' as unknown as string, isValidProfileName),
    'coder'
  )
})

test('readLegacyActiveProfile returns undefined when present but invalid (special chars)', () => {
  assert.equal(
    readLegacyActiveProfile('/p', () => 'BAD NAME!' as unknown as string, isValidProfileName),
    undefined
  )
})

test('readLegacyActiveProfile rejects default as a legacy CLI choice', () => {
  // "default" matches PROFILE_NAME_RE but is implicit; the explicit guard
  // suppresses it so the heuristic rung still fires.
  assert.equal(
    readLegacyActiveProfile('/p', () => 'default' as unknown as string, isValidProfileName),
    undefined
  )
})

test('readLegacyActiveProfile returns null for empty/whitespace content', () => {
  assert.equal(
    readLegacyActiveProfile('/p', () => '   \n' as unknown as string, isValidProfileName),
    null
  )
})

// ---------------------------------------------------------------------------
// findRunningGatewayProfiles
// ---------------------------------------------------------------------------

test('findRunningGatewayProfiles returns [] when no pid files exist', () => {
  const fs = makeFs({ '/home/u/.hermes/profiles/coder': { dir: true } })
  assert.deepEqual(
    findRunningGatewayProfiles('/home/u/.hermes/profiles', ['coder'], {
      ...fs,
      isHermesProcess: () => true
    }),
    []
  )
})

test('findRunningGatewayProfiles drops stale (non-hermes) recycled PIDs', () => {
  // Two profiles have pid files, but only coder's pid is a live hermes process.
  // The recycled PID at 5678 belongs to an unrelated process (e.g. Chrome).
  const fs = makeFs({
    '/home/u/.hermes/profiles/coder/gateway.pid': { content: '{"pid":1234}' },
    '/home/u/.hermes/profiles/writer/gateway.pid': { content: '{"pid":5678}' }
  })
  const deps = {
    ...fs,
    isHermesProcess: (pid: number) => pid === 1234
  }
  assert.deepEqual(
    findRunningGatewayProfiles('/home/u/.hermes/profiles', ['coder', 'writer'], deps),
    ['coder']
  )
})

test('findRunningGatewayProfiles tolerates malformed pid files', () => {
  const fs = makeFs({
    '/home/u/.hermes/profiles/coder/gateway.pid': { content: '{not json' }
  })
  assert.deepEqual(
    findRunningGatewayProfiles('/home/u/.hermes/profiles', ['coder'], {
      ...fs,
      isHermesProcess: () => true
    }),
    []
  )
})

test('findRunningGatewayProfiles drops non-integer and non-positive PIDs', () => {
  const fs = makeFs({
    '/home/u/.hermes/profiles/coder/gateway.pid': { content: '{"pid":-1}' },
    '/home/u/.hermes/profiles/writer/gateway.pid': { content: '{"pid":1.5}' },
    '/home/u/.hermes/profiles/extra/gateway.pid': { content: '{"pid":0}' }
  })
  assert.deepEqual(
    findRunningGatewayProfiles('/home/u/.hermes/profiles', ['coder', 'writer', 'extra'], {
      ...fs,
      isHermesProcess: () => true
    }),
    []
  )
})

test('findRunningGatewayProfiles preserves order of allProfiles', () => {
  const fs = makeFs({
    '/home/u/.hermes/profiles/coder/gateway.pid': { content: '{"pid":1}' },
    '/home/u/.hermes/profiles/writer/gateway.pid': { content: '{"pid":2}' }
  })
  const deps = { ...fs, isHermesProcess: () => true }
  assert.deepEqual(
    findRunningGatewayProfiles('/home/u/.hermes/profiles', ['coder', 'writer'], deps),
    ['coder', 'writer']
  )
})

// ---------------------------------------------------------------------------
// scoreStateDb
// ---------------------------------------------------------------------------

test('scoreStateDb returns null on missing file', () => {
  assert.equal(
    scoreStateDb('/missing', NOW, () => { throw new Error('ENOENT') }),
    null
  )
})

test('scoreStateDb floors recency weight at 0.1 for ancient files', () => {
  const ancient = scoreStateDb('/p', NOW, () => ({ size: 10 * 1024 * 1024, mtimeMs: 0 }))
  const fresh = scoreStateDb('/p', NOW, () => ({ size: 10 * 1024 * 1024, mtimeMs: NOW - 86_400_000 }))
  // Ancient still scores > 0 because recency is floored at 0.1.
  assert.ok(ancient !== null && ancient > 0)
  assert.ok(fresh !== null && fresh > ancient!)
})

test('scoreStateDb prefers larger DB at similar recency', () => {
  // 409 MB primary workspace vs 28 MB secondary — primary wins even with a
  // slightly newer mtime on the secondary.
  const big = scoreStateDb('/big', NOW, () => ({ size: 409 * 1024 * 1024, mtimeMs: NOW - 86_400_000 }))
  const small = scoreStateDb('/small', NOW, () => ({ size: 28 * 1024 * 1024, mtimeMs: NOW - 60_000 }))
  assert.ok(big !== null && small !== null && big > small)
})

test('scoreStateDb floors size weight at log10(MIN_SIZE) for tiny files', () => {
  // 100 bytes and PROFILE_SCORE_MIN_SIZE_BYTES score the same on the size axis
  // (recency is the same here, so the score is identical).
  const tiny = scoreStateDb('/tiny', NOW, () => ({ size: 100, mtimeMs: NOW - 86_400_000 }))
  const min = scoreStateDb('/min', NOW, () => ({ size: PROFILE_SCORE_MIN_SIZE_BYTES, mtimeMs: NOW - 86_400_000 }))
  assert.equal(tiny, min)
})

// ---------------------------------------------------------------------------
// listProfileDirs
// ---------------------------------------------------------------------------

test('listProfileDirs returns [] for missing profiles root', () => {
  const fs = makeFs({})
  assert.deepEqual(listProfileDirs(baseDeps({ ...fs })), [])
})

test('listProfileDirs includes default and any name passing the regex', () => {
  const fs = makeFs({
    '/home/u/.hermes/profiles/default': { dir: true },
    '/home/u/.hermes/profiles/coder': { dir: true },
    '/home/u/.hermes/profiles/writer': { dir: true }
  })
  assert.deepEqual(
    listProfileDirs(baseDeps({ ...fs })),
    ['default', 'coder', 'writer']
  )
})

test('listProfileDirs skips files (not directories) and invalid names', () => {
  const fs = makeFs({
    '/home/u/.hermes/profiles/default': { dir: true },
    '/home/u/.hermes/profiles/some-file.txt': { dir: false },
    '/home/u/.hermes/profiles/UPPERCASE': { dir: true } // regex rejects
  })
  assert.deepEqual(listProfileDirs(baseDeps({ ...fs })), ['default'])
})

// ---------------------------------------------------------------------------
// decideMigration
// ---------------------------------------------------------------------------

test('decideMigration prefers legacy over everything else', () => {
  const deps = baseDeps()
  const d = decideMigration('coder', ['writer'], ['coder', 'writer'], deps, () => null)
  assert.deepEqual(d, { profile: 'coder' })
  assert.equal(d?._migrated, undefined)
})

test('decideMigration prefers a single running gateway profile', () => {
  const deps = baseDeps()
  const d = decideMigration(null, ['coder'], ['coder', 'writer'], deps, () => null)
  assert.deepEqual(d, { profile: 'coder' })
  assert.equal(d?._migrated, undefined)
})

test('decideMigration falls through to scoring when multiple gateways run', () => {
  // When running.length > 1 the heuristic must score the running set, not all
  // profiles, so a stopped profile can't win against two live ones.
  const deps = baseDeps()
  const d = decideMigration(null, ['coder', 'writer'], ['coder', 'writer'], deps, p =>
    p.endsWith('/writer/state.db') ? 50 : 10
  )
  assert.deepEqual(d, { profile: 'writer', _migrated: true })
})

test('decideMigration returns null when no candidate scores and legacy is invalid', () => {
  const deps = baseDeps()
  assert.equal(
    decideMigration(undefined, [], ['coder', 'writer'], deps, () => null),
    null
  )
})

test('decideMigration suppresses write when best is default (single-profile fallback)', () => {
  // The whole point of the migration is to migrate AWAY from default when a
  // better candidate exists. If 'default' wins the score, the install is
  // single-profile and we leave it alone.
  const deps = baseDeps()
  const d = decideMigration(null, [], ['default', 'coder'], deps, p =>
    p.endsWith('/default/state.db') ? 99 : 50
  )
  assert.equal(d, null)
})

test('decideMigration still flags _migrated when legacy is invalid (undefined) but a heuristic wins', () => {
  // legacy === undefined means "file was present but malformed" — fall through to
  // the heuristic and mark as migrated.
  const deps = baseDeps()
  const d = decideMigration(undefined, [], ['coder'], deps, () => 10)
  assert.deepEqual(d, { profile: 'coder', _migrated: true })
})

// ---------------------------------------------------------------------------
// migrateActiveProfileIfMissing (orchestrator)
// ---------------------------------------------------------------------------

test('migrateActiveProfileIfMissing is a no-op when the preference file exists', () => {
  let written: unknown = null
  const deps = baseDeps({
    existsSync: (p: string) => p === '/cfg/active-profile.json',
    writeJson: (_p: string, payload: unknown) => { written = payload }
  })
  assert.equal(migrateActiveProfileIfMissing('/cfg/active-profile.json', deps), false)
  assert.equal(written, null)
})

test('migrateActiveProfileIfMissing writes legacy choice with no _migrated flag', () => {
  let written: unknown = null
  const fs = makeFs({
    '/home/u/.hermes/active_profile': { content: 'coder' },
    '/home/u/.hermes/profiles/coder': { dir: true },
    '/home/u/.hermes/profiles/writer': { dir: true }
  })
  const deps = baseDeps({
    ...fs,
    writeJson: (_p: string, payload: unknown) => { written = payload }
  })
  assert.equal(migrateActiveProfileIfMissing('/cfg/active-profile.json', deps), true)
  assert.deepEqual(written, { profile: 'coder' })
})

test('migrateActiveProfileIfMissing writes heuristic choice with _migrated=true', () => {
  let written: unknown = null
  const fs = makeFs({
    '/home/u/.hermes/profiles/coder/state.db': { size: 50 * 1024 * 1024, mtime: NOW - 86_400_000 },
    '/home/u/.hermes/profiles/writer/state.db': { size: 200 * 1024 * 1024, mtime: NOW - 86_400_000 }
  })
  const deps = baseDeps({
    ...fs,
    writeJson: (_p: string, payload: unknown) => { written = payload }
  })
  assert.equal(migrateActiveProfileIfMissing('/cfg/active-profile.json', deps), true)
  assert.deepEqual(written, { profile: 'writer', _migrated: true })
})

test('migrateActiveProfileIfMissing is a no-op for single-profile (default-only) installs', () => {
  // No heuristic candidate can beat 'default', so the orchestrator must NOT
  // write a file — preserves legacy launch behavior for the 99% case.
  let written: unknown = null
  const fs = makeFs({
    '/home/u/.hermes/profiles/default/state.db': { size: 10 * 1024 * 1024, mtime: NOW - 86_400_000 }
  })
  const deps = baseDeps({
    ...fs,
    writeJson: (_p: string, payload: unknown) => { written = payload }
  })
  assert.equal(migrateActiveProfileIfMissing('/cfg/active-profile.json', deps), false)
  assert.equal(written, null)
})

test('migrateActiveProfileIfMissing is a no-op when no profiles directory exists', () => {
  let written: unknown = null
  const fs = makeFs({})
  const deps = baseDeps({
    ...fs,
    writeJson: (_p: string, payload: unknown) => { written = payload }
  })
  assert.equal(migrateActiveProfileIfMissing('/cfg/active-profile.json', deps), false)
  assert.equal(written, null)
})

test('migrateActiveProfileIfMissing prefers a single running gateway over heuristics', () => {
  // running=['coder'] regardless of what the heuristic would score — gateway
  // ownership is the strongest signal we have for the active profile.
  let written: unknown = null
  const fs = makeFs({
    '/home/u/.hermes/profiles/coder/gateway.pid': { content: '{"pid":42}' },
    '/home/u/.hermes/profiles/writer/state.db': { size: 500 * 1024 * 1024, mtime: NOW - 86_400_000 }
  })
  const deps = baseDeps({
    ...fs,
    isHermesProcess: (pid: number) => pid === 42,
    writeJson: (_p: string, payload: unknown) => { written = payload }
  })
  assert.equal(migrateActiveProfileIfMissing('/cfg/active-profile.json', deps), true)
  assert.deepEqual(written, { profile: 'coder' })
})