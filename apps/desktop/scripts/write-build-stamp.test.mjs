import assert from 'node:assert/strict'
import { test } from 'vitest'

import {
  FALLBACK_BRANCH,
  FALLBACK_COMMIT,
  fromCI,
  deriveVersionMetadata,
  fromFallback,
  fromLocalGit,
  isFallbackCommit,
  resolveStamp
} from './write-build-stamp.mjs'

test('fromCI reads GITHUB_SHA / GITHUB_REF_NAME', () => {
  assert.deepEqual(
    fromCI({ GITHUB_SHA: 'a'.repeat(40), GITHUB_REF_NAME: 'release' }),
    { commit: 'a'.repeat(40), branch: 'release', dirty: false, source: 'ci' }
  )
  assert.equal(fromCI({}), null)
})

test('fromLocalGit returns null when git rev-parse fails', () => {
  const stamp = fromLocalGit('/tmp/not-a-repo', () => null)
  assert.equal(stamp, null)
})

test('fromLocalGit reads HEAD + branch + dirty status', () => {
  const calls = []
  const execFn = (cmd) => {
    calls.push(cmd)
    if (cmd === 'git rev-parse HEAD') return 'b'.repeat(40)
    if (cmd === 'git rev-parse --abbrev-ref HEAD') return 'main'
    if (cmd === 'git status --porcelain -uno') return ' M apps/desktop/package.json'
    return null
  }
  assert.deepEqual(fromLocalGit('/repo', execFn), {
    commit: 'b'.repeat(40),
    branch: 'main',
    dirty: true,
    source: 'local'
  })
  assert.ok(calls.includes('git rev-parse HEAD'))
})

test('fromFallback uses the all-zero placeholder commit', () => {
  assert.deepEqual(fromFallback(), {
    commit: FALLBACK_COMMIT,
    branch: FALLBACK_BRANCH,
    dirty: false,
    source: 'fallback'
  })
  assert.equal(isFallbackCommit(FALLBACK_COMMIT), true)
  assert.equal(isFallbackCommit('a'.repeat(40)), false)
})

test('resolveStamp prefers CI over local git over fallback', () => {
  const ci = resolveStamp({
    env: { GITHUB_SHA: 'c'.repeat(40), GITHUB_REF_NAME: 'main' },
    execFn: () => 'should-not-run'
  })
  assert.equal(ci.source, 'ci')
  assert.equal(ci.commit, 'c'.repeat(40))

  const local = resolveStamp({
    env: {},
    execFn: (cmd) => {
      if (cmd === 'git rev-parse HEAD') return 'd'.repeat(40)
      if (cmd === 'git rev-parse --abbrev-ref HEAD') return 'main'
      if (cmd === 'git status --porcelain -uno') return ''
      return null
    }
  })
  assert.equal(local.source, 'local')
  assert.equal(local.commit, 'd'.repeat(40))
  assert.equal(local.dirty, false)
})

test('resolveStamp falls back when neither CI nor git is available', () => {
  const stamp = resolveStamp({ env: {}, execFn: () => null })
  assert.deepEqual(stamp, {
    commit: FALLBACK_COMMIT,
    branch: FALLBACK_BRANCH,
    dirty: false,
    source: 'fallback'
  })
})

test('deriveVersionMetadata prefers a strict SemVer tag over historical CalVer tags', () => {
  const stamp = deriveVersionMetadata(
    { commit: 'a'.repeat(40), branch: 'feature', dirty: true, source: 'local' },
    {
      readFile: () => '__version__ = "0.20.0"\n__release_date__ = "2026.7.20"\n',
      execFn: command => {
        if (command.startsWith('git tag --merged')) return 'v2026.7.20\nv0.19.0\n'
        if (command === 'git rev-list --count v0.19.0..HEAD') return '7'
        if (command === 'git rev-list --count v2026.7.20..HEAD') return '20'
        return null
      }
    }
  )

  assert.deepEqual(stamp, {
    commit: 'a'.repeat(40), branch: 'feature', dirty: true, source: 'local',
    baseVersion: '0.19.0', displayVersion: '0.19.0+7', distance: 7
  })
})

test('deriveVersionMetadata uses the historical release tag only as a transition fallback', () => {
  const stamp = deriveVersionMetadata(
    { commit: 'a'.repeat(40), branch: 'feature', dirty: false, source: 'local' },
    {
      readFile: () => '__version__ = "0.19.0"\n__release_date__ = "2026.7.20"\n',
      execFn: command => command === 'git rev-list --count v2026.7.20..HEAD' ? '3' : ''
    }
  )

  assert.equal(stamp.baseVersion, '0.19.0')
  assert.equal(stamp.displayVersion, '0.19.0+3')
  assert.equal(stamp.distance, 3)
})

test('deriveVersionMetadata accepts three-digit SemVer majors but rejects four-digit CalVer years', () => {
  const stamp = deriveVersionMetadata(
    { commit: 'a'.repeat(40), branch: 'feature', dirty: false, source: 'local' },
    {
      readFile: () => '__version__ = "999.1.2"\n__release_date__ = "2026.7.20"\n',
      execFn: command => {
        if (command.startsWith('git tag --merged')) return 'v2026.7.20\nv999.1.2\n'
        if (command === 'git rev-list --count v999.1.2..HEAD') return '4'
        if (command === 'git rev-list --count v2026.7.20..HEAD') return '8'
        return null
      }
    }
  )

  assert.equal(stamp.baseVersion, '999.1.2')
  assert.equal(stamp.displayVersion, '999.1.2+4')
  assert.equal(stamp.distance, 4)
})
