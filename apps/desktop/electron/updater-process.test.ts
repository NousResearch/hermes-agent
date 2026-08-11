import assert from 'node:assert/strict'
import type { SpawnOptions } from 'node:child_process'
import path from 'node:path'

import { test } from 'vitest'

import {
  MARKER_SELF_ADOPT_EPOCH_MS,
  resolveStagedUpdaterBinary,
  resolveUpdateScriptHandoff,
  spawnUpdaterProcess,
  stagedUpdaterSupportsPrewrittenMarker,
  wrapHandoffForDetachedConsole
} from './updater-process'

const DAY_MS = 24 * 60 * 60 * 1000

test('stagedUpdaterSupportsPrewrittenMarker rejects installers predating the self-adopt fix', () => {
  // The real-world trap: an installer staged at first install months ago, never
  // refreshed because copy_self_to_hermes_home no-ops during --update.
  assert.equal(
    stagedUpdaterSupportsPrewrittenMarker('C:\\Hermes\\hermes-setup.exe', {
      stagedMtimeMs: () => MARKER_SELF_ADOPT_EPOCH_MS - 60 * DAY_MS
    }),
    false
  )
})

test('stagedUpdaterSupportsPrewrittenMarker accepts installers from the fix onward', () => {
  assert.equal(
    stagedUpdaterSupportsPrewrittenMarker('C:\\Hermes\\hermes-setup.exe', {
      stagedMtimeMs: () => MARKER_SELF_ADOPT_EPOCH_MS
    }),
    true
  )
  assert.equal(
    stagedUpdaterSupportsPrewrittenMarker('C:\\Hermes\\hermes-setup.exe', {
      stagedMtimeMs: () => MARKER_SELF_ADOPT_EPOCH_MS + 30 * DAY_MS
    }),
    true
  )
})

test('stagedUpdaterSupportsPrewrittenMarker treats an unreadable mtime as unsupported', () => {
  // Bias toward the path that can always make progress: a skipped pre-write
  // loses anti-respawn hardening, a wedged updater can never update again.
  assert.equal(
    stagedUpdaterSupportsPrewrittenMarker('C:\\Hermes\\hermes-setup.exe', {
      stagedMtimeMs: () => null
    }),
    false
  )
})

test('resolveStagedUpdaterBinary still returns a stale staged updater on Windows', () => {
  // Staleness gates only the marker PRE-WRITE, never the hand-off itself:
  // the stale binary is the only updater these users have, and it works fine
  // once it is allowed to write its own claim.
  assert.equal(
    resolveStagedUpdaterBinary('C:\\Hermes', {
      fileExists: () => true,
      isWindows: true,
      stagedMtimeMs: () => MARKER_SELF_ADOPT_EPOCH_MS - 60 * DAY_MS
    }),
    path.join('C:\\Hermes', 'hermes-setup.exe')
  )
})

test('spawnUpdaterProcess hides the updater console and detaches the child on Windows', () => {
  const calls: Array<{ args: string[]; command: string; options: SpawnOptions }> = []
  let unrefCalls = 0

  const child = {
    pid: 4242,
    unref: () => {
      unrefCalls += 1
    }
  }

  const result = spawnUpdaterProcess(
    'hermes-setup.exe',
    ['--update', '--branch', 'main'],
    { cwd: 'C:\\Hermes', detached: true, stdio: 'ignore' },
    {
      isWindows: true,
      spawnProcess: (command, args, options) => {
        calls.push({ args, command, options })

        return child
      }
    }
  )

  assert.equal(result, child)
  assert.equal(unrefCalls, 1)
  assert.deepEqual(calls, [
    {
      args: ['--update', '--branch', 'main'],
      command: 'hermes-setup.exe',
      options: { cwd: 'C:\\Hermes', detached: true, stdio: 'ignore', windowsHide: true }
    }
  ])
})

test('spawnUpdaterProcess preserves updater options off Windows', () => {
  let capturedOptions: SpawnOptions | undefined

  spawnUpdaterProcess(
    'hermes-setup',
    ['--update'],
    { detached: true, stdio: 'ignore' },
    {
      isWindows: false,
      spawnProcess: (_command, _args, options) => {
        capturedOptions = options

        return { unref: () => {} }
      }
    }
  )

  assert.deepEqual(capturedOptions, { detached: true, stdio: 'ignore' })
})

test('resolveStagedUpdaterBinary hands Windows the staged installer it finds', () => {
  const home = 'C:\\Users\\hermes\\AppData\\Local\\hermes'
  const staged = path.join(home, 'hermes-setup.exe')
  const probed: string[] = []

  const resolved = resolveStagedUpdaterBinary(home, {
    fileExists: candidate => {
      probed.push(candidate)

      return candidate === staged
    },
    isWindows: true
  })

  assert.equal(resolved, staged)
  assert.deepEqual(probed, [staged])
})

test('resolveStagedUpdaterBinary returns null off Windows even when hermes-setup is staged (#74836)', () => {
  const home = '/Users/hermes/.hermes'
  let probes = 0

  const resolved = resolveStagedUpdaterBinary(home, {
    // The installer stages hermes-setup on macOS/Linux too, so "it exists" is
    // the normal case — and precisely the one that must not win.
    fileExists: () => {
      probes += 1

      return true
    },
    isWindows: false
  })

  assert.equal(resolved, null)
  assert.equal(probes, 0)
})

test('resolveStagedUpdaterBinary returns null on Windows when nothing is staged', () => {
  const resolved = resolveStagedUpdaterBinary('C:\\Users\\hermes\\AppData\\Local\\hermes', {
    fileExists: () => false,
    isWindows: true
  })

  assert.equal(resolved, null)
})

test('resolveUpdateScriptHandoff prefers the repo script on Windows when present', () => {
  const root = String.raw`C:\Users\hermes\AppData\Local\hermes\hermes-agent`
  const expected = path.join(root, 'scripts', 'desktop-update', 'windows.ps1')

  const handoff = resolveUpdateScriptHandoff(root, {
    isWindows: true,
    fileExists: candidate => candidate === expected
  })

  assert.ok(handoff)
  assert.equal(handoff.command, 'powershell')
  assert.equal(handoff.scriptPath, expected)
  assert.deepEqual(handoff.args, ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', expected])
})

test('resolveUpdateScriptHandoff falls back to the pre-reorg flat path', () => {
  const root = String.raw`C:\Users\hermes\AppData\Local\hermes\hermes-agent`
  const legacy = path.join(root, 'scripts', 'desktop-update.ps1')

  const handoff = resolveUpdateScriptHandoff(root, {
    isWindows: true,
    fileExists: candidate => candidate === legacy
  })

  assert.ok(handoff)
  assert.equal(handoff.scriptPath, legacy)
})

test('resolveUpdateScriptHandoff returns null when the checkout predates the script', () => {
  const handoff = resolveUpdateScriptHandoff(String.raw`C:\Users\hermes\AppData\Local\hermes\hermes-agent`, {
    isWindows: true,
    fileExists: () => false
  })

  assert.equal(handoff, null)
})

test('resolveUpdateScriptHandoff is Windows-only (POSIX updates in place)', () => {
  const handoff = resolveUpdateScriptHandoff('/home/hermes/.hermes/hermes-agent', {
    isWindows: false,
    fileExists: () => true
  })

  assert.equal(handoff, null)
})

test('wrapHandoffForDetachedConsole routes through cmd start with own console', () => {
  const root = String.raw`C:\Users\hermes\AppData\Local\hermes\hermes-agent`
  const expected = path.join(root, 'scripts', 'desktop-update', 'windows.ps1')

  const handoff = resolveUpdateScriptHandoff(root, {
    isWindows: true,
    fileExists: candidate => candidate === expected
  })

  assert.ok(handoff)
  const wrapped = wrapHandoffForDetachedConsole(handoff, ['-InstallRoot', root, '-Branch', 'main'])

  assert.equal(wrapped.command, 'cmd.exe')
  assert.deepEqual(wrapped.args, [
    '/d',
    '/s',
    '/c',
    'start',
    '',
    '/min',
    'powershell',
    '-NoProfile',
    '-ExecutionPolicy',
    'Bypass',
    '-File',
    expected,
    '-InstallRoot',
    root,
    '-Branch',
    'main'
  ])
})
