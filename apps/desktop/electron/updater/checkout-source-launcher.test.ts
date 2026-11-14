import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { resolveInstallationLauncherAsync } from '../updater-process'

import { readSourceUpdate } from './checkout-source'

// Every window that mounts runs the passive update check on Electron's main
// thread (startUpdatePoller -> hermes:updates:check -> readSourceUpdate). On a
// PM checkout published only through an earlier user-bin launcher, proving that
// launcher's identity runs its `--version`, which can take seconds. The check
// must not block the event loop while it does, and must not pay it again.

let base: string

beforeEach((): void => {
  base = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'source-launcher-')))
  // The platform default home's bin is a launcher rung too: point it at
  // nothing so this test can never run a real installation's launcher.
  vi.stubEnv('HERMES_DATA_DIR_SUFFIX', `-test-${path.basename(base)}`)
})

afterEach((): void => {
  vi.unstubAllEnvs()
  fs.rmSync(base, { recursive: true, force: true })
})

/** A user-bin launcher that sleeps ~2 s on `--version`, then reports `root`; any other call prints JSON null. */
function writeSlowLauncher(home: string, root: string, calls: string): string {
  const bin: string = path.join(home, 'bin')
  fs.mkdirSync(bin, { recursive: true })

  if (process.platform === 'win32') {
    const launcher: string = path.join(bin, 'hermes.cmd')
    fs.writeFileSync(
      launcher,
      [
        '@echo off',
        `echo %~1>>"${calls}"`,
        'if "%~1"=="--version" (',
        '  ping -n 3 127.0.0.1 >nul',
        `  echo Install directory: ${root}`,
        '  exit /b 0',
        ')',
        'echo null',
        ''
      ].join('\r\n')
    )

    return launcher
  }

  const launcher: string = path.join(bin, 'hermes')
  fs.writeFileSync(
    launcher,
    `#!/bin/sh\necho "$1" >> '${calls}'\nif [ "$1" = "--version" ]; then sleep 2; printf 'Install directory: %s\\n' '${root}'; exit 0; fi\necho null\n`,
    { mode: 0o755 }
  )

  return launcher
}

it('proves a user-bin launcher again after it is replaced in place to target another checkout', async (): Promise<void> => {
  const root: string = path.join(base, 'checkout')
  const other: string = path.join(base, 'other-checkout')
  const home: string = path.join(base, 'home')
  const calls: string = path.join(base, 'calls')
  fs.mkdirSync(path.join(root, 'hermes_cli'), { recursive: true })
  fs.mkdirSync(path.join(root, 'pm'))
  fs.mkdirSync(other)
  fs.writeFileSync(path.join(root, 'hermes_cli', '_launchers.py'), '')

  const launcher: string = writeSlowLauncher(home, root, calls)

  expect(await resolveInstallationLauncherAsync(root, process.platform === 'win32', home)).toBe(launcher)

  writeSlowLauncher(home, other, calls)

  expect(await resolveInstallationLauncherAsync(root, process.platform === 'win32', home)).toBeNull()
}, 30000)

/** Longest gap between 10 ms ticks while `work` runs: how long the event loop was blocked. */
async function longestBlock<T>(work: () => Promise<T>): Promise<{ result: T; blockedMs: number }> {
  let last: number = performance.now()
  let blockedMs = 0

  const timer = setInterval((): void => {
    const now: number = performance.now()
    blockedMs = Math.max(blockedMs, now - last)
    last = now
  }, 10)

  try {
    const result: T = await work()
    blockedMs = Math.max(blockedMs, performance.now() - last)

    return { result, blockedMs }
  } finally {
    clearInterval(timer)
  }
}

it('resolves a slow user-bin launcher without blocking the event loop, once per process', async (): Promise<void> => {
  const root: string = path.join(base, 'checkout')
  const home: string = path.join(base, 'home')
  const calls: string = path.join(base, 'calls')
  fs.mkdirSync(path.join(root, 'hermes_cli'), { recursive: true })
  fs.mkdirSync(path.join(root, 'pm'))
  fs.writeFileSync(path.join(root, 'hermes_cli', '_launchers.py'), '')
  writeSlowLauncher(home, root, calls)

  const probe = { python: null, git: 'git', updateRoot: root, hermesHome: home }

  const versionProbes = (): number =>
    fs
      .readFileSync(calls, 'utf8')
      .split(/\r?\n/)
      .filter((line: string): boolean => line.trim() === '--version').length

  const first = await longestBlock(() => readSourceUpdate(probe))

  expect(first.result).toBeNull()
  expect(versionProbes()).toBe(1)
  // spawnSync held the loop for the whole ~2 s probe; an async probe only pays process creation.
  expect(first.blockedMs).toBeLessThan(100)

  // A second window's check reuses the proven identity: no second `--version`.
  const second = await longestBlock(() => readSourceUpdate(probe))

  expect(second.result).toBeNull()
  expect(versionProbes()).toBe(1)
  expect(second.blockedMs).toBeLessThan(100)
}, 30000)
