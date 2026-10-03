/**
 * Tests for electron/installs-cli.ts.
 *
 * These are the pure helpers behind the desktop `hermes installs` UI: argv
 * building, id validation, and JSON parsing of the CLI output.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  dismissArgs,
  listArgs,
  parseInstallsList,
  removeArgs,
  runTimeoutMs,
  shouldShowBootNotice,
  validateInstallId
} from './installs-cli'

// --- argv building ---

test('listArgs and dismissArgs use argv arrays, never a shell string', () => {
  assert.deepEqual(listArgs(), ['installs', 'list', '--json'])
  assert.deepEqual(dismissArgs(), ['installs', 'dismiss'])
})

test('remove may run longer than the CLI waits for Remove-AppxPackage (300 s)', () => {
  assert.equal(runTimeoutMs(listArgs()), 30_000)
  assert.equal(runTimeoutMs(dismissArgs()), 30_000)
  assert.ok(runTimeoutMs(['installs', 'remove', 'abc123', '--yes']) > 300_000)
})

test('removeArgs builds the confirmed-removal argv for a valid id', () => {
  assert.deepEqual(removeArgs('abc123'), ['installs', 'remove', 'abc123', '--yes'])
  assert.deepEqual(removeArgs(' abc123 '), ['installs', 'remove', 'abc123', '--yes'])
})

// --- validateInstallId ---

test('validateInstallId accepts short hex ids and rejects everything else', () => {
  assert.equal(validateInstallId('a'), 'a')
  assert.equal(validateInstallId('0123456789ab'), '0123456789ab')

  for (const bad of ['', ' ', 'ABC', 'g', '0123456789abc', 'a;b', 'a&b', 42, null, undefined, {}]) {
    assert.equal(validateInstallId(bad as unknown as string), null, String(bad))
  }
})

// --- parseInstallsList ---

const SAMPLE = {
  current: 'abc',
  installs: [
    {
      id: 'abc',
      root: '/home/u/hermes-agent',
      steward: 'git',
      version: '1.2.3',
      sources: ['path'],
      current: true,
      package_full_name: null,
      removable: false,
      action: 'refuse',
      refusal: 'not removed: this install is running'
    },
    {
      id: 'def9',
      root: 'C:/Apps/Hermes',
      steward: 'desktop-app',
      version: null,
      sources: ['appx'],
      current: false,
      package_full_name: 'NousLabs.Hermes_1.0.0.0_x64__pqr',
      removable: true,
      action: 'appx',
      refusal: null
    }
  ],
  launchers: [{ path: 'C:/Apps/Hermes/hermes.exe', owner: 'def9' }],
  notice: { count: 1, dismissed: false, partial: false }
}

test('parseInstallsList reads the JSON from noisy stdout', () => {
  const parsed = parseInstallsList(`warning: whatever\n${JSON.stringify(SAMPLE, null, 2)}\n`)

  assert.ok(parsed)
  assert.equal(parsed.current, 'abc')
  assert.equal(parsed.installs.length, 2)
  assert.equal(parsed.installs[0]?.removable, false)
  assert.equal(parsed.installs[1]?.package_full_name, 'NousLabs.Hermes_1.0.0.0_x64__pqr')
  assert.deepEqual(parsed.notice, { count: 1, dismissed: false, partial: false })
})

test('parseInstallsList returns null on malformed output', () => {
  assert.equal(parseInstallsList(''), null)
  assert.equal(parseInstallsList('not json'), null)
  assert.equal(parseInstallsList('{'), null)
  assert.equal(parseInstallsList('[1, 2]'), null)
  assert.equal(parseInstallsList('{"current": 5}'), null)
  assert.equal(parseInstallsList('{"current": "abc"}'), null)
})

test('parseInstallsList returns null on partial output', () => {
  assert.equal(parseInstallsList('{"current": "abc", "installs": []}'), null)
  assert.equal(
    parseInstallsList(
      '{"current": "abc", "installs": [{"id": "abc", "root": "/r", "steward": "git", "sources": [], "current": true, "package_full_name": null, "removable": false, "action": "refuse", "refusal": null}], "launchers": [], "notice": {"count": 0, "dismissed": true, "partial": false}}'
    ),
    null
  )
  assert.equal(
    parseInstallsList(
      '{"current": "abc", "installs": [], "launchers": [{"path": "p"}], "notice": {"count": 0, "dismissed": false, "partial": false}}'
    ),
    null
  )
  assert.equal(
    parseInstallsList(
      '{"current": "abc", "installs": [], "launchers": [], "notice": {"count": "two", "dismissed": false, "partial": false}}'
    ),
    null
  )
})

test('parseInstallsList rejects a notice without the partial flag', () => {
  assert.equal(
    parseInstallsList('{"current": "abc", "installs": [], "launchers": [], "notice": {"count": 0, "dismissed": false}}'),
    null
  )
})

test('parseInstallsList accepts an empty list with no launchers', () => {
  const parsed = parseInstallsList(
    '{"current": "abc", "installs": [], "launchers": [], "notice": {"count": 0, "dismissed": false, "partial": false}}'
  )

  assert.ok(parsed)
  assert.deepEqual(parsed.installs, [])
})

// --- shouldShowBootNotice ---

test('shouldShowBootNotice shows only for other installs that are not dismissed', () => {
  assert.equal(shouldShowBootNotice(null), false)
  assert.equal(shouldShowBootNotice({ notice: { count: 0, dismissed: false, partial: false } } as never), false)
  assert.equal(shouldShowBootNotice({ notice: { count: 2, dismissed: true, partial: false } } as never), false)
  assert.equal(shouldShowBootNotice({ notice: { count: 2, dismissed: false, partial: false } } as never), true)
})
