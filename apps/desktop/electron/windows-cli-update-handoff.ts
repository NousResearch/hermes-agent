/**
 * windows-cli-update-handoff.ts
 *
 * Pure helpers for the Windows Update path when the staged Tauri installer
 * (`HERMES_HOME/hermes-setup.exe`) is missing.
 *
 * Background (#46755 / #46779)
 * ---------------------------
 * The GUI Update button prefers handing off to the staged hermes-setup binary.
 * That binary is only written by a full bootstrap install. CLI-only installs
 * and some CLI-driven rebuild paths never stage it, so `resolveStagedUpdaterBinary`
 * returns null and the desktop used to surface a manual `hermes update` dialog
 * forever — even when a working venv `hermes` is right there.
 *
 * Preferred contract (issue Option B): when the staged binary is absent but a
 * hermes CLI is available, detach a small cmd watcher that:
 *   1. waits for the desktop PID to exit (releases the Windows venv shim lock),
 *   2. runs `hermes update --yes` (branch-pinned when needed),
 *   3. relaunches the desktop executable.
 *
 * Kept electron-free so vitest can cover the script shape without booting the
 * main process (same pattern as desktop-uninstall.ts / update-relaunch.ts).
 */

import fs from 'node:fs'
import path from 'node:path'

export interface ResolveHermesCliBinaryDeps {
  fileExists?: (candidate: string) => boolean
  findOnPath?: (command: string) => string | null
  isWindows?: boolean
}

/**
 * Resolve a hermes CLI that can drive `hermes update` for the install at
 * `updateRoot`. Prefer the venv shim inside that checkout; fall back to PATH.
 *
 * Windows venv layout is `venv/Scripts/hermes.exe` (not `venv/bin/hermes`).
 * Looking only at the POSIX path made Windows CLI handoff always miss the
 * local venv and depend on a PATH install.
 */
export function resolveHermesCliBinary(updateRoot: string, deps: ResolveHermesCliBinaryDeps = {}): string | null {
  if (!updateRoot) {
    return null
  }

  const isWindows = deps.isWindows ?? process.platform === 'win32'
  const pathMod = isWindows ? path.win32 : path.posix

  const fileExists =
    deps.fileExists ??
    ((candidate: string) => {
      try {
        return fs.statSync(candidate).isFile()
      } catch {
        return false
      }
    })

  const findOnPath = deps.findOnPath ?? (() => null)

  const candidates = isWindows
    ? [
        pathMod.join(updateRoot, 'venv', 'Scripts', 'hermes.exe'),
        pathMod.join(updateRoot, 'venv', 'Scripts', 'hermes.cmd'),
        pathMod.join(updateRoot, 'venv', 'Scripts', 'hermes.bat')
      ]
    : [pathMod.join(updateRoot, 'venv', 'bin', 'hermes')]

  for (const candidate of candidates) {
    if (fileExists(candidate)) {
      return candidate
    }
  }

  return findOnPath('hermes') || null
}

export interface BuildWindowsCliUpdateScriptOpts {
  desktopPid: number
  hermesCmd: string
  updateRoot: string
  hermesHome: string
  /** Branch to pin, or null/empty/'main' for bare `hermes update`. */
  branch?: string | null
  /** Desktop executable to relaunch after a successful update. */
  relaunchExe?: string | null
  /** Extra args to pass the relaunched desktop (already filtered). */
  relaunchArgs?: string[]
}

/**
 * Build a cmd.exe watcher script for the no-staged-updater Windows path.
 *
 * Mirrors the wait/relaunch shape used by desktop-uninstall's Windows cleanup
 * script: bounded PID wait, then work, then self-delete.
 */
export function buildWindowsCliUpdateScript(opts: BuildWindowsCliUpdateScriptOpts): string {
  const pid = Number(opts.desktopPid) || 0
  // cmd.exe has no real string escaping inside quotes; strip embedded quotes
  // (Hermes install paths under %LOCALAPPDATA% never contain them).
  const q = (s: string) => `"${String(s).replace(/"/g, '')}"`
  const branch = (opts.branch || '').trim()
  const branchArgs = branch && branch !== 'main' ? ` --branch ${branch.replace(/[^A-Za-z0-9._/-]/g, '')}` : ''

  const lines = [
    '@echo off',
    'setlocal enableextensions',
    `set "HERMES_HOME=${String(opts.hermesHome).replace(/"/g, '')}"`,
    `set "PID=${pid}"`,
    'set /a waited=0',
    ':waitloop',
    'rem Exact PID wait (same pattern as desktop-uninstall Windows cleanup).',
    'tasklist /NH /FI "PID eq %PID%" 2>nul | findstr /r /c:" %PID% " >nul',
    'if %ERRORLEVEL% neq 0 goto waited_done',
    'set /a waited+=1',
    'if %waited% geq 60 goto waited_done',
    'timeout /t 1 /nobreak >nul',
    'goto waitloop',
    ':waited_done',
    `cd /d ${q(opts.updateRoot)}`,
    // hermes update already rebuilds desktop when a packaged app is present.
    `${q(opts.hermesCmd)} update --yes${branchArgs}`
  ]

  if (opts.relaunchExe) {
    const args = (opts.relaunchArgs || []).map(a => q(a)).join(' ')
    // `start ""` is required so start does not treat the first quoted token
    // as a window title.
    lines.push(args ? `start "" ${q(opts.relaunchExe)} ${args}` : `start "" ${q(opts.relaunchExe)}`)
  }

  lines.push('del "%~f0"')
  lines.push('')

  return lines.join('\r\n')
}

/**
 * Decide whether Windows should try the CLI handoff instead of the manual
 * dialog. Pure boolean for tests.
 */
export function shouldUseWindowsCliUpdateHandoff(opts: {
  isWindows: boolean
  stagedUpdater: string | null | undefined
  hermesCli: string | null | undefined
}): boolean {
  return Boolean(opts.isWindows && !opts.stagedUpdater && opts.hermesCli)
}
