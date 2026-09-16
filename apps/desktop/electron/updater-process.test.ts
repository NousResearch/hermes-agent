import assert from 'node:assert/strict'
import { type SpawnOptions, spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  collectRelaunchArgs,
  describeUpdaterHandoffFailure,
  launchWindowsUpdateTransport,
  observeUpdaterHandoff,
  resolvePosixScriptHandoff,
  resolveStagedUpdaterBinary,
  resolveUpdateScriptHandoff,
  resolveWindowsDevRelaunchAppPath,
  resolveWindowsUpdateTransport,
  sandboxFallbackFromEnv,
  spawnUpdaterProcess,
  WINDOWS_HANDOFF_ENV,
  type WindowsUpdateTransport,
  wrapHandoffForDetachedConsole
} from './updater-process'

test('dev relaunch uses Electron resolved app path independently of launch switches', () => {
  const appPath = 'C:\\Hermes proof\\apps\\desktop'
  assert.equal(resolveWindowsDevRelaunchAppPath(true, appPath), appPath)
  assert.equal(resolveWindowsDevRelaunchAppPath(false, appPath), undefined)
})


test('resolveStagedUpdaterBinary still returns a stale staged updater on Windows', () => {
  // Staleness gates only the marker PRE-WRITE, never the hand-off itself:
  // the stale binary is the only updater these users have, and it works fine
  // once it is allowed to write its own claim.
  assert.equal(
    resolveStagedUpdaterBinary('C:\\Hermes', {
      fileExists: () => true,
      isWindows: true
    }),
    path.join('C:\\Hermes', 'hermes-setup.exe')
  )
})

function scriptTransport(root: string): WindowsUpdateTransport & { kind: 'script' } {
  const expected = path.join(root, 'scripts', 'desktop-update', 'windows.ps1')
  const handoff = resolveUpdateScriptHandoff(root, { isWindows: true, fileExists: candidate => candidate === expected })

  assert.ok(handoff)

  return { kind: 'script', handoff }
}

const HANDOFF_VALUES = {
  branch: 'main',
  desktopPid: 42,
  installRoot: String.raw`C:\Users\hermes\AppData\Local\hermes\hermes-agent`,
  nonce: 'a'.repeat(48),
  relaunchExe: String.raw`C:\Hermes\Hermes.exe`
}

test('the Windows script hand-off spawns its cmd wrapper non-detached even when the caller asks for detached', () => {
  // #116161: a DETACHED_PROCESS cmd.exe owns no console, so `start /b` would
  // leave powershell to allocate a new visible one (or die in console init).
  const calls: SpawnOptions[] = []

  const launch = launchWindowsUpdateTransport(
    scriptTransport(HANDOFF_VALUES.installRoot),
    HANDOFF_VALUES,
    { cwd: String.raw`C:\Hermes`, detached: true, stdio: 'ignore', env: { KEEP: '1' } },
    {
      isWindows: true,
      spawnProcess: (_command, _args, options) => {
        calls.push(options)

        return { pid: 7, unref: () => {} }
      }
    }
  )

  assert.equal(launch.kind, 'spawned')
  assert.equal(calls.length, 1)
  assert.equal(calls[0].detached, false)
  assert.equal(calls[0].windowsHide, true)
  assert.equal(calls[0].env?.KEEP, '1', 'caller env survives')
  assert.equal(calls[0].env?.[WINDOWS_HANDOFF_ENV.nonce], HANDOFF_VALUES.nonce)
})

test('a remote-served hand-off tells the script not to restart a local gateway; a local one does not', () => {
  // #117529: a remote-served Desktop must not (re)start a local messaging
  // gateway that competes with the remote host's channel polling.
  const { handoff } = scriptTransport(HANDOFF_VALUES.installRoot)
  const remote = wrapHandoffForDetachedConsole(handoff, { ...HANDOFF_VALUES, noGateway: true })
  const local = wrapHandoffForDetachedConsole(handoff, HANDOFF_VALUES)

  assert.equal(remote.env[WINDOWS_HANDOFF_ENV.noGateway], '1')
  assert.equal(WINDOWS_HANDOFF_ENV.noGateway in local.env, false)
})

test.skipIf(process.platform !== 'win32')(
  'the encoded launcher binds the no-gateway flag onto the script -NoGateway switch',
  () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-handoff-launcher-'))

    try {
      const out = path.join(dir, 'bound.txt')
      const script = path.join(dir, 'windows.ps1')

      fs.writeFileSync(
        script,
        [
          'param([string]$InstallRoot, [string]$Branch, [int]$DesktopPid, [string]$RelaunchExe,',
          '      [string]$HandoffNonce, [string]$RelaunchAppPath, [switch]$NoGateway)',
          `Set-Content -LiteralPath '${out.replace(/'/g, "''")}' -Value ([string][bool]$NoGateway)`,
          'exit 0'
        ].join('\r\n')
      )

      const run = (noGateway: boolean) => {
        const wrapped = wrapHandoffForDetachedConsole(
          { command: 'powershell.exe', args: [], scriptPath: script },
          { ...HANDOFF_VALUES, installRoot: dir, noGateway }
        )

        // args[6..] is `<powershell> -NoProfile ... -EncodedCommand <launcher>`;
        // run it directly (no cmd `start`) so the exit is synchronous.
        const result = spawnSync(wrapped.args[6], wrapped.args.slice(7), {
          env: { ...process.env, ...wrapped.env },
          encoding: 'utf8',
          windowsHide: true
        })

        assert.equal(result.status, 0, String(result.stderr || result.stdout))

        return fs.readFileSync(out, 'utf8').trim()
      }

      assert.equal(run(true), 'True')
      assert.equal(run(false), 'False')
    } finally {
      fs.rmSync(dir, { recursive: true, force: true })
    }
  },
  60_000
)

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

test('resolveWindowsUpdateTransport selects the live checkout script', () => {
  const root = String.raw`C:\Users\hermes\AppData\Local\hermes\hermes-agent`
  const scriptPath = path.join(root, 'scripts', 'desktop-update', 'windows.ps1')

  const transport = resolveWindowsUpdateTransport(root, {
    isWindows: true,
    fileExists: candidate => candidate === scriptPath
  })

  assert.equal(transport.kind, 'script')
  assert.equal(transport.kind === 'script' ? transport.handoff.scriptPath : null, scriptPath)
})

test('resolveWindowsUpdateTransport requires a manual update without a live script', () => {
  const transport = resolveWindowsUpdateTransport(String.raw`C:\Users\hermes\AppData\Local\hermes\hermes-agent`, {
    isWindows: true,
    fileExists: () => false
  })

  assert.deepEqual(transport, { kind: 'manual' })
})

test('wrapHandoffForDetachedConsole runs the script inside a non-detached hidden wrapper console', () => {
  // #116161: `start /min` allocated a NEW (minimized, visible) console for
  // powershell on every hand-off; `detached: true` (DETACHED_PROCESS) would
  // leave the wrapper console-less, forcing the same allocation under `/b`.

  const root = String.raw`C:\Users\hermes\AppData\Local\hermes\hermes-agent`
  const expected = path.join(root, 'scripts', 'desktop-update', 'windows.ps1')

  const handoff = resolveUpdateScriptHandoff(root, {
    isWindows: true,
    fileExists: candidate => candidate === expected
  })

  assert.ok(handoff)
  const wrapped = wrapHandoffForDetachedConsole(handoff, ['-InstallRoot', root, '-Branch', 'main'])

  assert.equal(wrapped.command, 'cmd.exe')
  assert.equal(wrapped.detached, false)
  assert.deepEqual(wrapped.args, [
    '/d',
    '/s',
    '/c',
    'start',
    '',
    '/b',
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

test('authenticated Windows handoff uses the absolute inbox PowerShell path', () => {
  const root = String.raw`C:\Users\hermes\AppData\Local\hermes\hermes-agent`
  const expected = path.join(root, 'scripts', 'desktop-update', 'windows.ps1')

  const handoff = resolveUpdateScriptHandoff(root, {
    isWindows: true,
    fileExists: candidate => candidate === expected
  })

  assert.ok(handoff)

  const wrapped = wrapHandoffForDetachedConsole(handoff, {
    branch: 'main',
    desktopPid: 42,
    installRoot: root,
    nonce: 'a'.repeat(48),
    relaunchExe: String.raw`C:\Hermes\Hermes.exe`
  })

  const powershell = path.join(
    process.env.SystemRoot || 'C:\\Windows',
    'System32',
    'WindowsPowerShell',
    'v1.0',
    'powershell.exe'
  )

  assert.equal(wrapped.command, 'cmd.exe')
  assert.equal(wrapped.args[6], powershell)
  assert.equal(wrapped.env?.HERMES_UPDATE_HANDOFF_SCRIPT, expected)
})

test('resolvePosixScriptHandoff returns the bash recipe when the script exists', () => {
  const root = '/home/hermes/.hermes/hermes-agent'
  const expected = path.join(root, 'scripts', 'desktop-update', 'posix.sh')

  const handoff = resolvePosixScriptHandoff(root, {
    isWindows: false,
    fileExists: candidate => candidate === expected
  })

  assert.ok(handoff)
  assert.equal(handoff.command, '/bin/bash')
  assert.deepEqual(handoff.args, [expected])
})

test('resolvePosixScriptHandoff is null when the checkout predates the script', () => {
  const handoff = resolvePosixScriptHandoff('/home/hermes/.hermes/hermes-agent', {
    isWindows: false,
    fileExists: () => false
  })

  assert.equal(handoff, null)
})

test('resolvePosixScriptHandoff is null on Windows', () => {
  const handoff = resolvePosixScriptHandoff(String.raw`C:\Users\hermes\AppData\Local\hermes\hermes-agent`, {
    isWindows: true,
    fileExists: () => true
  })

  assert.equal(handoff, null)
})

test('collectRelaunchArgs drops Electron internals, keeps user/launcher args', () => {
  const argv = [
    '--type=renderer',
    '--user-data-dir=/tmp/x',
    '--enable-features=A,B',
    '--field-trial-handle=123',
    '--enable-logging',
    '--log-file=/tmp/log',
    '--lang=en-US',
    '--inspect=9229',
    '--remote-debugging-port=9222',
    '--no-sandbox',
    'hermes://open/session/abc',
    '--profile=work'
  ]

  assert.deepEqual(collectRelaunchArgs(argv), ['--no-sandbox', 'hermes://open/session/abc', '--profile=work'])
  assert.deepEqual(collectRelaunchArgs(undefined), [])
})

test('sandboxFallbackFromEnv: ELECTRON_DISABLE_SANDBOX / --no-sandbox opt out', () => {
  assert.equal(sandboxFallbackFromEnv({ ELECTRON_DISABLE_SANDBOX: '1' }, []), true)
  assert.equal(sandboxFallbackFromEnv({ ELECTRON_DISABLE_SANDBOX: 'true' }, []), true)
  assert.equal(sandboxFallbackFromEnv({}, ['--no-sandbox']), true)
  assert.equal(sandboxFallbackFromEnv({ ELECTRON_DISABLE_SANDBOX: '0' }, []), false)
  assert.equal(sandboxFallbackFromEnv({}, []), false)
})

// ── observeUpdaterHandoff (#66753) ──────────────────────────────────────────

class FakeChild {
  pid = 1234
  listeners = new Map<string, Array<(...args: unknown[]) => void>>()
  removed: string[] = []

  unref() {}

  once(event: string, listener: (...args: unknown[]) => void) {
    const arr = this.listeners.get(event) ?? []

    arr.push(listener)
    this.listeners.set(event, arr)

    return this
  }

  removeListener(event: string, _listener: (...args: unknown[]) => void) {
    this.removed.push(event)

    return this
  }

  emit(event: string, ...args: unknown[]) {
    for (const listener of this.listeners.get(event) ?? []) {
      listener(...args)
    }
  }
}

function manualTimer() {
  const pending: Array<() => void> = []

  return {
    deps: {
      setTimeoutFn: (callback: () => void, _ms: number) => {
        pending.push(callback)

        return 0
      },
      clearTimeoutFn: () => {}
    },
    fire: () => {
      for (const callback of pending.splice(0)) {
        callback()
      }
    }
  }
}

test('observeUpdaterHandoff reports a spawn error instead of settling ok', async () => {
  const child = new FakeChild()
  const timer = manualTimer()
  const outcomePromise = observeUpdaterHandoff(child, 2500, timer.deps)

  const err: Error & { code?: string } = new Error('spawn ENOENT')

  err.code = 'ENOENT'
  child.emit('error', err)

  const outcome = await outcomePromise

  assert.equal(outcome.ok, false)
  assert.equal(outcome.reason, 'spawn-error')
  assert.match(outcome.message ?? '', /ENOENT/)
})

test('observeUpdaterHandoff reports a non-zero early exit', async () => {
  const child = new FakeChild()
  const timer = manualTimer()
  const outcomePromise = observeUpdaterHandoff(child, 2500, timer.deps)

  child.emit('exit', 127, null)

  const outcome = await outcomePromise

  assert.equal(outcome.ok, false)
  assert.equal(outcome.reason, 'early-exit')
  assert.equal(outcome.code, 127)
})

test('describeUpdaterHandoffFailure leads with plain copy and confines the raw outcome to Details', () => {
  for (const raw of ['updater exited 127 before the settle window elapsed', 'updater spawn failed: ENOENT']) {
    const text = describeUpdaterHandoffFailure({ message: raw })
    const [lead, details] = text.split('\n\nDetails: ')

    assert.ok(lead && !lead.includes(raw))
    assert.equal(details, raw)
  }

  assert.doesNotMatch(describeUpdaterHandoffFailure({}), /Details:/)
})

test('observeUpdaterHandoff reports a signal death inside the window', async () => {
  const child = new FakeChild()
  const timer = manualTimer()
  const outcomePromise = observeUpdaterHandoff(child, 2500, timer.deps)

  child.emit('exit', null, 'SIGTERM')

  const outcome = await outcomePromise

  assert.equal(outcome.ok, false)
  assert.equal(outcome.reason, 'early-exit')
  assert.equal(outcome.signal, 'SIGTERM')
})

test('observeUpdaterHandoff accepts a clean exit 0 (Windows cmd start wrapper)', async () => {
  const child = new FakeChild()
  const timer = manualTimer()
  const outcomePromise = observeUpdaterHandoff(child, 2500, timer.deps)

  child.emit('exit', 0, null)

  const outcome = await outcomePromise

  assert.equal(outcome.ok, true)
  assert.equal(outcome.code, 0)
})

test('observeUpdaterHandoff settles ok when the child survives the window', async () => {
  const child = new FakeChild()
  const timer = manualTimer()
  const outcomePromise = observeUpdaterHandoff(child, 2500, timer.deps)

  timer.fire()

  const outcome = await outcomePromise

  assert.equal(outcome.ok, true)
  assert.equal(outcome.reason, undefined)
  // Listeners must be detached so a post-quit late exit can't fire them.
  assert.deepEqual(child.removed.sort(), ['error', 'exit'])
})

test('observeUpdaterHandoff ignores events after the first settle', async () => {
  const child = new FakeChild()
  const timer = manualTimer()
  const outcomePromise = observeUpdaterHandoff(child, 2500, timer.deps)

  child.emit('exit', 1, null)
  child.emit('error', new Error('late'))
  timer.fire()

  const outcome = await outcomePromise

  assert.equal(outcome.ok, false)
  assert.equal(outcome.reason, 'early-exit')
})

test('observeUpdaterHandoff settles ok for children without an event interface', async () => {
  const timer = manualTimer()
  const outcomePromise = observeUpdaterHandoff({ pid: 1, unref: () => {} }, 2500, timer.deps)

  timer.fire()

  const outcome = await outcomePromise

  assert.equal(outcome.ok, true)
})
