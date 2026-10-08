/**
 * The checkout script's marker helper (A7 rule 3, SPEC section 6): protocol
 * negotiation from the script text, the exact command line, and the one-line
 * verdict — exercised against a REAL fake script process (bash), with the
 * spawn injected only to pin the Windows command line on a POSIX host.
 */

import fs from 'fs'
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import os from 'os'
import path from 'path'

import { afterEach, describe, test } from 'vitest'

import {
  checkoutLockMayBeHeld,
  checkoutLockPath,
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
    ['foreign', { kind: 'foreign' }],
    ['busy', { kind: 'busy' }],
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
    assert.equal(
      fs.readFileSync(path.join(home, 'helper-calls.log'), 'utf8'),
      `--marker-op reclaim --install-root ${root}\n`
    )
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

  test('a verdict only the other op can give is an error, never clearance (R8 m7)', async () => {
    const { root, home } = fakeHelperCheckout()
    const opts = { updateRoot: root, hermesHome: home, desktopPid: 4242, runId: 'desk-4242-abc-0001', isWindows: false }

    for (const [op, verdict, expected] of [
      ['reclaim', 'live 31337', 'live'],
      ['reclaim', 'held', 'held'],
      ['reclaim', 'reclaimed', 'reclaimed'],
      ['reclaim', 'foreign', 'error'],
      ['reclaim', 'withdrawn', 'error'],
      ['reclaim', 'taken 99', 'error'],
      ['withdraw', 'held', 'error'],
      ['withdraw', 'reclaimed', 'error'],
      ['withdraw', 'live 31337', 'error']
    ] as const) {
      fs.writeFileSync(path.join(home, 'helper-verdict'), verdict)
      assert.equal((await runMarkerHelper(op, opts)).kind, expected, `${op} => ${verdict}`)
    }
  })

  test.skipIf(process.getuid?.() === 0)(
    'a script that exists but cannot be read is an error, never an older checkout (R8 M5)',
    async () => {
      const { root, home } = fakeHelperCheckout()
      fs.writeFileSync(path.join(home, 'helper-verdict'), 'reclaimed')
      fs.chmodSync(path.join(root, 'scripts', 'desktop-update', 'posix.sh'), 0o000)

      assert.deepEqual(await runMarkerHelper('reclaim', { updateRoot: root, hermesHome: home, isWindows: false }), {
        kind: 'error'
      })
    }
  )

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

// Review K132354 (helper always spawned): with no marker the gate skips the
// helper only when the script would answer "not held" for a missing checkout
// lock, so the TS path must be the script's own.
describe.skipIf(process.platform === 'win32')('checkout lock path mirrors marker.sh', () => {
  const markerSh = path.resolve(__dirname, '../../../../scripts/desktop-update/marker.sh')

  const scriptPath = (root: string) =>
    execFileSync('bash', ['-c', 'source "$0"; checkout_lock_path', markerSh], {
      env: { ...process.env, INSTALL_ROOT: root },
      encoding: 'utf8'
    }).trim()

  const layouts: Record<string, (root: string, other: string) => void> = {
    'no .git (ZIP install)': () => {},
    '.git directory': root => fs.mkdirSync(path.join(root, '.git')),
    'linked worktree: BOM, CRLF, relative gitdir, relative commondir': root => {
      fs.mkdirSync(path.join(root, 'wt', 'admin'), { recursive: true })
      fs.writeFileSync(path.join(root, '.git'), '\uFEFFgitdir: wt/admin \r\n')
      fs.writeFileSync(path.join(root, 'wt', 'admin', 'commondir'), '../common\n')
    },
    'absolute gitdir and commondir': (root, other) => {
      fs.writeFileSync(path.join(root, '.git'), `gitdir:${other}\n`)
      fs.writeFileSync(path.join(other, 'commondir'), `\t${path.join(other, 'main')}\n`)
    },
    '.git file that is not a gitdir pointer': root => fs.writeFileSync(path.join(root, '.git'), 'GITDIR: x\n'),
    'empty .git file': root => fs.writeFileSync(path.join(root, '.git'), ''),
    'empty commondir': root => {
      fs.mkdirSync(path.join(root, '.git'))
      fs.writeFileSync(path.join(root, '.git', 'commondir'), '\n')
    }
  }

  test.each(Object.keys(layouts))('%s', name => {
    const root = scratchRoot('lock-path')
    const other = scratchRoot('lock-path-other')
    layouts[name](root, other)

    const expected = scriptPath(root)
    assert.equal(path.resolve(checkoutLockPath(root, false)), path.resolve(expected))
    assert.equal(checkoutLockMayBeHeld(root, false), false, 'no lock file: the script answers not held')
    fs.mkdirSync(path.dirname(expected), { recursive: true })
    fs.writeFileSync(expected, '')
    assert.equal(checkoutLockMayBeHeld(root, false), true, 'a lock file: only the script can tell')
  })

  test('an unreadable .git pointer asks the script', () => {
    const root = scratchRoot('lock-path-unreadable')
    fs.mkdirSync(path.join(root, '.git', 'x'), { recursive: true })
    fs.rmSync(path.join(root, '.git'), { recursive: true })
    fs.writeFileSync(path.join(root, '.git'), 'gitdir: elsewhere\n', { mode: 0o000 })

    if (process.getuid?.() === 0) {
      return // root reads it anyway
    }

    assert.equal(checkoutLockMayBeHeld(root, false), true)
  })
})
