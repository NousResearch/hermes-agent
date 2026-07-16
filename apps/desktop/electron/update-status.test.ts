/**
 * Tests for electron/update-status.ts — pure logic for interpreting
 * hermes-updater status output and routing the apply decision.
 *
 * Run with: npx vitest run electron/update-status.test.ts --environment jsdom
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  interpretUpdaterStatus,
  resolveInstallType,
  routeApplyDecision,
  type InstallType,
  type UpdaterStatusJson
} from './update-status'

// ---------------------------------------------------------------------------
// interpretUpdaterStatus
// ---------------------------------------------------------------------------

test('interpretUpdaterStatus: update available maps to supported + behind + commits', () => {
  const json: UpdaterStatusJson = {
    current_version: '1.42.0',
    latest_version: '1.43.0',
    update_available: true,
    behind: 3,
    channel: 'stable',
    current_sha: 'abc123',
    target_sha: 'def456',
    changelog: [
      { version: '1.43.0', summary: 'Release 1.43', author: 'alice', at: 1700000000, sha: 'def456' },
      { version: '1.42.1', summary: 'Hotfix', author: 'bob', at: 1699000000, sha: 'bbb222' },
      { version: '1.42.0', summary: 'Initial', author: 'carol', at: 1698000000, sha: 'abc123' }
    ]
  }

  const result = interpretUpdaterStatus(json, { fetchedAt: 12345 })

  assert.equal(result.supported, true)
  assert.equal(result.updateAvailable, true)
  assert.equal(result.behind, 3)
  assert.equal(result.branch, 'stable')
  assert.equal(result.currentSha, 'abc123')
  assert.equal(result.targetSha, 'def456')
  assert.equal(result.fetchedAt, 12345)
  assert.equal(result.commits!.length, 3)
  assert.equal(result.commits![0].summary, 'Release 1.43')
  assert.equal(result.commits![0].author, 'alice')
  // Seconds → ms conversion (git convention)
  assert.equal(result.commits![0].at, 1700000000 * 1000)
})

test('interpretUpdaterStatus: up-to-date maps to behind 0 and no commits', () => {
  const json: UpdaterStatusJson = {
    current_version: '1.43.0',
    latest_version: '1.43.0',
    update_available: false,
    behind: 0,
    channel: 'stable'
  }

  const result = interpretUpdaterStatus(json)

  assert.equal(result.supported, true)
  assert.equal(result.updateAvailable, false)
  assert.equal(result.behind, 0)
  assert.deepEqual(result.commits, [])
  assert.equal(result.commits!.length, 0)
})

test('interpretUpdaterStatus: update_available true with no behind field defaults behind to 1', () => {
  const json: UpdaterStatusJson = {
    current_version: '1.42.0',
    latest_version: '1.43.0',
    update_available: true,
    channel: 'stable'
  }

  const result = interpretUpdaterStatus(json)

  assert.equal(result.updateAvailable, true)
  assert.equal(result.behind, 1, 'behind defaults to 1 when update_available and no explicit count')
})

test('interpretUpdaterStatus: updater error surfaces as fetch-failed (retryable)', () => {
  const json: UpdaterStatusJson = {
    error: 'connection refused',
    channel: 'stable'
  }

  const result = interpretUpdaterStatus(json)

  assert.equal(result.supported, true, 'the install is valid; only the check failed')
  assert.equal(result.error, 'fetch-failed')
  assert.equal(result.message, 'connection refused')
  assert.equal(result.behind, 0)
  assert.deepEqual(result.commits, [])
  assert.equal(result.branch, 'stable')
})

test('interpretUpdaterStatus: invalid JSON returns fetch-failed', () => {
  for (const bad of [null, 'not an object', [1, 2, 3], undefined]) {
    const result = interpretUpdaterStatus(bad)
    assert.equal(result.supported, true)
    assert.equal(result.error, 'fetch-failed')
    assert.equal(result.message, 'hermes-updater returned invalid JSON.')
    assert.equal(result.behind, 0)
    assert.deepEqual(result.commits, [])
    assert.equal(typeof result.fetchedAt, 'number')
  }
})

test('interpretUpdaterStatus: changelog timestamps in ms are not doubled', () => {
  const msTimestamp = 1700000000000 // already ms
  const json: UpdaterStatusJson = {
    update_available: true,
    behind: 1,
    changelog: [{ summary: 'v2', at: msTimestamp }]
  }

  const result = interpretUpdaterStatus(json)

  assert.equal(result.commits![0].at, msTimestamp, 'ms timestamps pass through unchanged')
})

test('interpretUpdaterStatus: missing changelog results in empty commits array', () => {
  const json: UpdaterStatusJson = {
    update_available: true,
    behind: 2,
    channel: 'beta'
  }

  const result = interpretUpdaterStatus(json)

  assert.deepEqual(result.commits, [])
})

test('interpretUpdaterStatus: empty changelog array results in empty commits', () => {
  const json: UpdaterStatusJson = {
    update_available: true,
    behind: 1,
    changelog: []
  }

  const result = interpretUpdaterStatus(json)

  assert.deepEqual(result.commits, [])
})

test('interpretUpdaterStatus: changelog entries with missing fields are tolerated', () => {
  const json: UpdaterStatusJson = {
    update_available: true,
    behind: 1,
    changelog: [{ version: '1.0' }]  // missing sha, summary, author, at
  }

  const result = interpretUpdaterStatus(json)

  assert.equal(result.commits!.length, 1)
  assert.equal(result.commits![0].sha, '')
  assert.equal(result.commits![0].summary, '')
  assert.equal(result.commits![0].author, '')
  assert.equal(result.commits![0].at, 0)
})

test('interpretUpdaterStatus: fetchedAt defaults to Date.now() when not provided', () => {
  const before = Date.now()
  const result = interpretUpdaterStatus({ update_available: false })
  const after = Date.now()

  assert.ok(result.fetchedAt! >= before, 'fetchedAt should be >= call time')
  assert.ok(result.fetchedAt! <= after, 'fetchedAt should be <= call time')
})

// ---------------------------------------------------------------------------
// routeApplyDecision
// ---------------------------------------------------------------------------

test('routeApplyDecision: slot routes to updater handoff', () => {
  const route = routeApplyDecision('slot')

  assert.equal(route.route, 'updater')
  assert.equal(route.command, 'hermes-updater')
  assert.ok(route.args.includes('apply'), 'args include the apply verb')
  assert.ok(route.args.includes('--relaunch-app'), 'args include --relaunch-app')
  assert.ok(route.args.includes('{execPath}'), 'args include the execPath placeholder')
  assert.ok(route.args.includes('--report'), 'args include --report')
  assert.ok(route.args.includes('json'), 'args include json')
  assert.ok(route.args.includes('--notify-file'), 'args include --notify-file')
  assert.ok(route.args.includes('{notifyFile}'), 'args include the notifyFile placeholder')
})

test('routeApplyDecision: checkout routes to dev-update (hermes update)', () => {
  const route = routeApplyDecision('checkout')

  assert.equal(route.route, 'dev-update')
  assert.equal(route.command, 'hermes')
  assert.ok(route.args.includes('update'), 'args include update')
  assert.ok(route.args.includes('--yes'), 'args include --yes')
})

test('routeApplyDecision: package routes to gui-skew with message', () => {
  const route = routeApplyDecision('package')

  assert.equal(route.route, 'gui-skew')
  if (route.route === 'gui-skew') {
    assert.ok(route.message.length > 0, 'gui-skew has a message')
    assert.ok(
      route.message.toLowerCase().includes('package') || route.message.toLowerCase().includes('reinstall'),
      'message mentions package or reinstall'
    )
  }
})

test('routeApplyDecision: all install types are handled (exhaustive)', () => {
  const types: InstallType[] = ['slot', 'checkout', 'package']

  for (const t of types) {
    const route = routeApplyDecision(t)
    assert.ok(['updater', 'dev-update', 'gui-skew'].includes(route.route), `${t} has a valid route`)
  }
})

// ---------------------------------------------------------------------------
// resolveInstallType
// ---------------------------------------------------------------------------

test('resolveInstallType: versions/ + current.txt → slot', () => {
  const probes = {
    directoryExists: (p: string) => p.endsWith('versions'),
    fileExists: (p: string) => p.endsWith('current.txt')
  }

  assert.equal(resolveInstallType('/home/u/.hermes', '/home/u/.hermes/hermes-agent', probes), 'slot')
})

test('resolveInstallType: .git dir in active root → checkout', () => {
  const probes = {
    directoryExists: (p: string) => p.endsWith('.git'),
    fileExists: (_p: string) => false
  }

  assert.equal(resolveInstallType('/home/u/.hermes', '/home/u/.hermes/hermes-agent', probes), 'checkout')
})

test('resolveInstallType: .git file (worktree) in active root → checkout', () => {
  const probes = {
    directoryExists: (_p: string) => false,
    fileExists: (p: string) => p.endsWith('.git')
  }

  assert.equal(resolveInstallType('/home/u/.hermes', '/home/u/.hermes/hermes-agent', probes), 'checkout')
})

test('resolveInstallType: neither versions/ nor .git → package', () => {
  const probes = {
    directoryExists: (_p: string) => false,
    fileExists: (_p: string) => false
  }

  assert.equal(resolveInstallType('/home/u/.hermes', '/home/u/.hermes/hermes-agent', probes), 'package')
})

test('resolveInstallType: slot takes precedence over checkout', () => {
  // If both versions/ + current.txt exist AND .git exists (mid-migration),
  // the slot layout wins — the install has been adopted into managed mode.
  const probes = {
    directoryExists: (_p: string) => true,
    fileExists: (_p: string) => true
  }

  assert.equal(resolveInstallType('/home/u/.hermes', '/home/u/.hermes/hermes-agent', probes), 'slot')
})


