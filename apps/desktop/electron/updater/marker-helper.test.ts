/**
 * The checkout script's marker helper (A7 rule 3, SPEC section 6): protocol
 * negotiation from the script text, the exact command line, and the one-line
 * verdict — exercised against a REAL fake script process (bash), with the
 * spawn injected only to pin the Windows command line on a POSIX host.
 */

import fs from 'fs'
import assert from 'node:assert/strict'
import os from 'os'
import path from 'path'

import { afterEach, describe, test } from 'vitest'

import {
  type HelperSpawn,
  markerHelperCommand,
  parseMarkerHelperVerdict,
  readHandoffProtocol,
  runMarkerHelper
} from './marker-helper'
import { cleanupFakeCheckouts, fakeHelperCheckout, scratchRoot } from './marker-helper.test-helpers'

afterEach(cleanupFakeCheckouts)

describe('protocol negotiation', () => {
  test.each([
    ['exact protocol 2 line', '#!/bin/sh\n# hermes-handoff-protocol: 2\n', 2],
    ['trailing whitespace', '# hermes-handoff-protocol: 2  \r\n', 2],
    ['a later protocol', 'x\n# hermes-handoff-protocol: 3\n', 3],
    ['protocol 1 is legacy', '# hermes-handoff-protocol: 1\n', 1],
    ['no line (origin/main scripts)', '#!/usr/bin/env bash\necho hi\n', 1],
    ['indented line does not count', '  # hermes-handoff-protocol: 2\n', 1],
    ['mid-line mention does not count', 'echo "# hermes-handoff-protocol: 2"\n', 1],
    ['two spaces after the colon', '# hermes-handoff-protocol:  2\n', 1]
  ])('%s', (_label, text, expected) => {
    const file = path.join(fakeHelperCheckout().root, 'script.sh')
    fs.writeFileSync(file, text)

    assert.equal(readHandoffProtocol(file), expected)
  })

  test('a missing script is legacy', () => {
    assert.equal(readHandoffProtocol(path.join(os.tmpdir(), 'no-such-script.sh')), 1)
    assert.equal(readHandoffProtocol(null), 1)
  })
})

test.each([
  ['absent', { kind: 'absent' }],
  ['reclaimed\n', { kind: 'reclaimed' }],
  ['held', { kind: 'held' }],
  ['busy', { kind: 'busy' }],
  ['withdrawn\r\n', { kind: 'withdrawn' }],
  ['foreign', { kind: 'foreign' }],
  ['live 4242', { kind: 'live', pid: 4242 }],
  ['taken 77\n', { kind: 'taken', pid: 77 }],
  ['live', { kind: 'error' }],
  ['taken x', { kind: 'error' }],
  ['taken 0', { kind: 'error' }],
  ['held 5', { kind: 'error' }],
  ['reclaimed\nheld', { kind: 'error' }],
  ['usage: posix.sh [--branch B]', { kind: 'error' }],
  ['', { kind: 'error' }]
])('verdict line %j', (stdout, expected) => {
  assert.deepEqual(parseMarkerHelperVerdict(stdout), expected)
})

describe.skipIf(process.platform === 'win32')('runMarkerHelper against a real script process', () => {
  test.each([
    ['reclaimed', { kind: 'reclaimed' }],
    ['held', { kind: 'held' }],
    ['live 31337', { kind: 'live', pid: 31337 }],
    ['taken 99', { kind: 'taken', pid: 99 }],
    ['withdrawn', { kind: 'withdrawn' }]
  ])('prints %j', async (verdict, expected) => {
    const { root, home } = fakeHelperCheckout()
    fs.writeFileSync(path.join(home, 'helper-verdict'), verdict)

    const got = await runMarkerHelper('withdraw', {
      updateRoot: root,
      hermesHome: home,
      desktopPid: 4242,
      runId: 'desk-4242-abc-0001',
      isWindows: false
    })

    assert.deepEqual(got, expected)
    assert.equal(
      fs.readFileSync(path.join(home, 'helper-calls.log'), 'utf8'),
      `--marker-op withdraw --install-root ${root} --desktop-pid 4242 --handoff-run desk-4242-abc-0001\n`,
      'exact argv, and HERMES_HOME reached the script'
    )
  })

  test('reclaim without a desktop pid or run passes neither flag', async () => {
    const { root, home } = fakeHelperCheckout()
    fs.writeFileSync(path.join(home, 'helper-verdict'), 'absent')

    assert.deepEqual(await runMarkerHelper('reclaim', { updateRoot: root, hermesHome: home, isWindows: false }), {
      kind: 'absent'
    })
    assert.equal(fs.readFileSync(path.join(home, 'helper-calls.log'), 'utf8'), `--marker-op reclaim --install-root ${root}\n`)
  })

  test.each([
    ['protocol-2 helper exits 64', '64', 'reclaimed'],
    ['nonzero exit even with a verdict', '1', 'reclaimed'],
    ['garbage output', '0', 'sure, done'],
    ['two lines', '0', 'reclaimed\nheld']
  ])('%s => error', async (_label, exit, verdict) => {
    const { root, home } = fakeHelperCheckout()
    fs.writeFileSync(path.join(home, 'helper-verdict'), verdict)
    fs.writeFileSync(path.join(home, 'helper-exit'), exit)

    assert.deepEqual(await runMarkerHelper('reclaim', { updateRoot: root, hermesHome: home, isWindows: false }), {
      kind: 'error'
    })
  })

  test('a helper past its timeout is killed and reports error', async () => {
    const { root, home } = fakeHelperCheckout()
    fs.writeFileSync(path.join(home, 'helper-verdict'), 'reclaimed')
    fs.writeFileSync(path.join(home, 'helper-sleep'), '5')
    const started = Date.now()

    assert.deepEqual(
      await runMarkerHelper('reclaim', { updateRoot: root, hermesHome: home, isWindows: false, timeoutMs: 300 }),
      { kind: 'error' }
    )
    assert.ok(Date.now() - started < 4000, 'bounded by the timeout')
  })

  test('a checkout without the script is unsupported', async () => {
    const root = scratchRoot('marker-helper-none')

    assert.deepEqual(await runMarkerHelper('reclaim', { updateRoot: root, hermesHome: root, isWindows: false }), {
      kind: 'unsupported'
    })
  })
})

test('the Windows command line: powershell -File windows.ps1 -MarkerOp ... (hidden, HERMES_HOME set)', async () => {
  const root = scratchRoot('marker-helper-win')
  const script = path.join(root, 'scripts', 'desktop-update', 'windows.ps1')
  fs.mkdirSync(path.dirname(script), { recursive: true })
  fs.writeFileSync(script, '# hermes-handoff-protocol: 2\n')
  const calls: Parameters<HelperSpawn>[] = []

  const spawn: HelperSpawn = async (...args) => {
    calls.push(args)

    return { code: 0, stdout: 'taken 4040\r\n' }
  }

  const got = await runMarkerHelper('withdraw', {
    updateRoot: root,
    hermesHome: path.join(root, 'home'),
    desktopPid: 12,
    runId: 'desk-12-a-beef',
    isWindows: true,
    spawn
  })

  assert.deepEqual(got, { kind: 'taken', pid: 4040 })
  assert.equal(calls.length, 1)
  const [command, args, options] = calls[0]
  assert.equal(command, 'powershell.exe')
  assert.deepEqual(args, [
    '-NoProfile',
    '-NonInteractive',
    '-ExecutionPolicy',
    'Bypass',
    '-File',
    script,
    '-MarkerOp',
    'withdraw',
    '-InstallRoot',
    root,
    '-DesktopPid',
    '12',
    '-HandoffRun',
    'desk-12-a-beef'
  ])
  assert.equal(options.env.HERMES_HOME, path.join(root, 'home'))
  assert.equal(options.windowsHide, true)
  assert.equal(options.timeout, 20_000)
  assert.deepEqual(markerHelperCommand('reclaim', script, { updateRoot: root, isWindows: true }).args.slice(-4), [
    '-MarkerOp',
    'reclaim',
    '-InstallRoot',
    root
  ])
})
