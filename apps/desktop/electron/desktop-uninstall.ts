/**
 * desktop-uninstall.ts
 *
 * Pure, electron-free helpers for the desktop Chat GUI uninstaller. These map
 * the three user-facing uninstall modes to the `hermes uninstall` CLI flags,
 * resolve the running app bundle/exe so a detached cleanup script can remove
 * it after the app quits, and build that cleanup script for each OS.
 *
 * Kept electron-free so Vitest can exercise the registered IPC handlers.
 * main.ts supplies the local install stamp and process callbacks.
 *
 * The three modes mirror the CLI's options exactly:
 *   - 'gui'  → remove ONLY the Chat GUI, keep the agent + all user data.
 *              `hermes uninstall --gui --yes`
 *   - 'lite' → remove the GUI + agent code, KEEP user data (config / sessions
 *              / .env) for a future reinstall. `hermes uninstall --yes`
 *   - 'full' → remove everything: GUI + agent + all user data.
 *              `hermes uninstall --full --yes`
 *
 * Why a detached cleanup script: 'lite'/'full' delete the very venv the
 * `hermes` command runs from, and every mode may need to delete the running
 * app bundle (locked on macOS/Windows while the process is alive). So we hand
 * the work to a detached child that waits for this app's PID to exit, runs the
 * Python uninstall, then removes the app bundle — then the app quits. Same
 * shape as the self-update swap-and-relaunch flow already in main.ts.
 */

import path from 'node:path'

import type { InstallStamp } from './install-stamp'

export interface UninstallSummaryDetails {
  hermes_home: string
  agent_installed: boolean
  gui_installed: boolean
  source_built_artifacts: string[]
  packaged_app_paths: string[]
  userdata_dir: string
  userdata_exists: boolean
  platform: string
  running_app_path?: string | null
  probe?: string
}

export interface DesktopUninstallSummary extends UninstallSummaryDetails {
  code_removal_allowed: boolean
}

export interface DesktopUninstallResult {
  ok: boolean
  mode?: string
  willRemoveAppBundle?: boolean
  scriptPath?: string
  error?: string
  message?: string
}

export interface DesktopUninstallIpcDeps {
  ipcMain: {
    handle: (channel: string, handler: (event: unknown, payload?: unknown) => Promise<unknown>) => void
  }
  stamp: Readonly<Partial<Pick<InstallStamp, 'distribution' | 'source' | 'payload' | 'updateMechanism'>>> | null
  fallbackSummary: () => UninstallSummaryDetails
  probeSummary: () => Promise<UninstallSummaryDetails>
  runUninstall: (mode: string) => Promise<DesktopUninstallResult>
}

export function registerDesktopUninstallIpc({
  ipcMain,
  stamp,
  fallbackSummary,
  probeSummary,
  runUninstall
}: DesktopUninstallIpcDeps): void {
  const kind: InstallKind = resolveInstallKind(stamp ?? {})
  const codeRemovalAllowed: boolean = installKindAllowsCodeRemoval(kind)

  ipcMain.handle('hermes:uninstall:summary', async (): Promise<DesktopUninstallSummary> => {
    const summary: UninstallSummaryDetails = codeRemovalAllowed ? await probeSummary() : fallbackSummary()

    // The local artifact owns this decision, not the Python summary.
    return { ...summary, code_removal_allowed: codeRemovalAllowed }
  })
  ipcMain.handle(
    'hermes:uninstall:run',
    async (_event: unknown, payload?: unknown): Promise<DesktopUninstallResult> => {
      // Every cleanup mode can remove the bundle, including a hidden data request.
      if (!codeRemovalAllowed) {
        return {
          ok: false,
          error: 'externally-managed',
          message: 'This desktop install must be removed through its installer or package manager.'
        }
      }

      const mode: unknown = payload && typeof payload === 'object' && 'mode' in payload ? payload.mode : payload
      const requestedMode: string = String(mode || '')

      if (!allowedUninstallModes(kind).includes(requestedMode)) {
        return { ok: false, error: 'invalid-mode', message: `Unknown uninstall mode: ${requestedMode}` }
      }

      return runUninstall(requestedMode)
    }
  )
}

const UNINSTALL_MODES: string[] = ['gui', 'lite', 'full', 'data']

// The baked install stamp determines who owns removal:
//   'nix'      — a Nix build (stamp distribution 'nix'). The store is
//                immutable and the install is owned by Nix tooling, so the
//                app must not remove any code, its own bundle included.
//   'bundled'  — a bundled or light artifact. The OS owns app removal.
//   'external' — another package manager owns updates and removal.
//   'standard' — everything else: the git-clone install the desktop
//                installer bootstraps, or a `hermes desktop` source build.
//                The classic script flow (venv python + rm the bundle) works.
//
// Only 'standard' installs may use the desktop cleanup script.
const INSTALL_KINDS = ['nix', 'bundled', 'external', 'standard'] as const
type InstallKind = (typeof INSTALL_KINDS)[number]

/**
 * Classify the install from the stamp. Pure so it can be unit-tested:
 * callers pass the baked stamp, never the connected backend's install facts.
 * `distribution` is authoritative; `source` is the schema-1 fallback.
 * The 'bundled' and 'light' artifact kinds both classify as the managed
 * 'bundled' flow — neither has agent code the app may remove, and the OS
 * owns app removal.
 */
function resolveInstallKind({
  distribution,
  source,
  payload = 'bootstrap',
  updateMechanism
}: NonNullable<DesktopUninstallIpcDeps['stamp']> = {}): InstallKind {
  if (distribution === 'nix' || source === 'nix') {
    return 'nix'
  }

  if (payload === 'bundled' || payload === 'light') {
    return 'bundled'
  }

  if (updateMechanism === 'external') {
    return 'external'
  }

  return 'standard'
}

/** True when this install kind lets the app remove code (agent / bundle). */
function installKindAllowsCodeRemoval(kind: InstallKind): boolean {
  return kind === 'standard'
}

/**
 * Desktop cleanup always removes the bundle. No mode is safe for an install
 * owned by another installer, including the CLI's data-only mode.
 */
function allowedUninstallModes(kind: InstallKind): string[] {
  return installKindAllowsCodeRemoval(kind) ? ['gui', 'lite', 'full'] : []
}

/**
 * Human instructions for removing the app itself the native way. Used when
 * the install kind forbids code removal: Windows owns the bundled app
 * through Apps & Features, macOS through the Trash, a Linux AppImage is a
 * single file the user placed somewhere, and a Nix install belongs to the
 * flake / profile that made it. `appPath` is the resolveRemovableAppPath()
 * result (the AppImage path on Linux), used only to name the exact file.
 */
function nativeRemovalInstructions(kind, platform, appPath = null) {
  if (kind === 'nix') {
    return (
      'This Hermes desktop app was installed by Nix. Uninstall it the same way you installed it: ' +
      'remove hermes-agent from your flake or profile, then rebuild.'
    )
  }

  if (platform === 'win32') {
    return 'To uninstall, go to Windows Settings → Apps → Installed apps.'
  }

  if (platform === 'darwin') {
    return 'Quit the app and drag Hermes.app from Applications to the Trash.'
  }

  if (appPath && /\.appimage$/i.test(String(appPath))) {
    return `Delete the AppImage file at ${appPath}.`
  }

  if (appPath) {
    return `Delete the app directory at ${appPath}.`
  }

  return 'Delete the Hermes AppImage (or app directory) from wherever you saved it.'
}

/**
 * Map an uninstall mode to the `python -m hermes_cli.uninstall` argv (after the
 * python executable). Uses the dedicated lightweight module entrypoint (not
 * `hermes_cli.main`) so it can run under a system Python OUTSIDE the venv that
 * lite/full delete — see the Finding-3 note in buildWindowsCleanupScript.
 * Throws on an unknown mode so a typo can't silently become a full wipe.
 */
function uninstallArgsForMode(mode: string) {
  if (!UNINSTALL_MODES.includes(mode)) {
    throw new Error(`Unknown uninstall mode: ${mode}`)
  }

  return ['-m', 'hermes_cli.uninstall', '--mode', mode]
}

/** True when `mode` removes the agent code (lite/full), false otherwise. */
function modeRemovesAgent(mode: string) {
  return mode === 'lite' || mode === 'full'
}

/** True when `mode` removes user data (full and data). */
function modeRemovesUserData(mode: string) {
  return mode === 'full' || mode === 'data'
}

/**
 * Resolve the on-disk app bundle/dir to remove for the running desktop app,
 * given the path to the running executable (`process.execPath`) and platform.
 *
 *   macOS:   …/Hermes.app/Contents/MacOS/Hermes  → …/Hermes.app
 *   Windows: …\Hermes\Hermes.exe                 → …\Hermes  (install dir)
 *   Linux:   AppImage → the APPIMAGE env path; unpacked → the *-unpacked dir
 *
 * Returns null when we can't confidently identify a removable bundle (e.g.
 * running from a dev checkout, or a system-package install we must not rmtree).
 */
function resolveRemovableAppPath(execPath, platform, env: any = {}) {
  const exe = String(execPath || '')

  if (!exe) {
    return null
  }

  // Use the path flavor that matches the TARGET platform, not the host running
  // this code — so the Windows branch parses backslash paths correctly even
  // when these pure helpers are unit-tested on Linux/macOS CI.
  const p = platform === 'win32' ? path.win32 : path.posix

  if (platform === 'darwin') {
    // …/Hermes.app/Contents/MacOS/Hermes → strip 3 segments to the .app
    const macOsDir = p.dirname(exe) // …/Contents/MacOS
    const contents = p.dirname(macOsDir) // …/Contents
    const appBundle = p.dirname(contents) // …/Hermes.app

    if (appBundle.endsWith('.app')) {
      return appBundle
    }

    return null
  }

  if (platform === 'win32') {
    // NSIS per-user installs Hermes.exe directly in the install dir.
    const dir = p.dirname(exe)

    if (/[\\/]Hermes$/i.test(dir) || /[\\/]hermes-desktop$/i.test(dir)) {
      return dir
    }

    return null
  }

  // Linux: an AppImage exposes its own path via the APPIMAGE env var.
  if (env.APPIMAGE) {
    return env.APPIMAGE
  }

  // Unpacked electron-builder tree: …/linux-unpacked/hermes
  const dir = p.dirname(exe)

  if (/-unpacked$/.test(dir)) {
    return dir
  }

  return null
}

/**
 * Should we even try to remove the running app bundle from a cleanup script?
 * Only when packaged AND we resolved a concrete removable path. Dev runs
 * (electron from node_modules) and system-package installs return null above
 * and are left to the OS package manager.
 */
function shouldRemoveAppBundle(isPackaged, appPath) {
  return Boolean(isPackaged) && Boolean(appPath)
}

/**
 * Build a POSIX cleanup shell script (macOS / Linux). It:
 *   1. waits (bounded ~30s) for the desktop PID to exit (venv/bundle unlock),
 *   2. runs the Python uninstall module with the mode,
 *   3. removes the app bundle if one was resolved.
 *
 * `pythonExe` should be a Python OUTSIDE the venv for lite/full (the venv is
 * being deleted); `pythonPath` is prepended to PYTHONPATH so `import hermes_cli`
 * resolves from the agent source. `q()` single-quote-escapes for the shell
 * (closes-escapes-reopens any embedded apostrophe), defending against spaces.
 */
function buildPosixCleanupScript({ desktopPid, pythonExe, pythonPath, agentRoot, uninstallArgs, appPath, hermesHome }) {
  const q = s => `'${String(s).replace(/'/g, `'\\''`)}'`

  const lines = [
    '#!/usr/bin/env bash',
    'set -u',
    '# Wait (up to ~30s) for the desktop process to exit so the venv python',
    '# and the app bundle are no longer in use.',
    `pid=${Number(desktopPid) || 0}`,
    'if [ "$pid" -gt 0 ]; then',
    '  for _ in $(seq 1 60); do',
    '    kill -0 "$pid" 2>/dev/null || break',
    '    sleep 0.5',
    '  done',
    'fi',
    `export HERMES_HOME=${q(hermesHome)}`
  ]

  if (pythonPath) {
    lines.push(`export PYTHONPATH=${q(pythonPath)}\${PYTHONPATH:+:$PYTHONPATH}`)
  }

  lines.push(`cd ${q(agentRoot)} 2>/dev/null || true`, `${q(pythonExe)} ${uninstallArgs.map(q).join(' ')} || true`)

  if (appPath) {
    lines.push(`rm -rf ${q(appPath)} || true`)
  }

  // Self-delete the script.
  lines.push('rm -f "$0" 2>/dev/null || true')
  lines.push('')

  return lines.join('\n')
}

/**
 * Build a Windows cleanup batch script. Same three steps, cmd.exe flavored.
 *
 * Finding 3 (venv self-deletion): for lite/full the agent uninstall rmtree's
 * the venv that contains `python.exe`. A running .exe is mandatory-locked on
 * Windows, so running the uninstall from the venv's OWN python half-fails. The
 * desktop passes a system Python (findSystemPython) as `pythonExe` for those
 * modes + `pythonPath`=agentRoot so `import hermes_cli` resolves from source
 * while the venv is torn down. gui-only doesn't touch the venv, so it can use
 * either interpreter.
 *
 * Wait-loop: bounded (matches POSIX's ~30s cap) so a never-exiting / mismatched
 * PID can't wedge the cleanup forever. The `/FI "PID eq"` filter is an EXACT
 * match, so no redundant `| find` (which would substring-match 99→990).
 *
 * Removal: even after the desktop PID is gone, Windows releases directory
 * handles lazily, so a single `rmdir /s /q` can half-fail — retry up to 10x.
 */
function buildWindowsCleanupScript({
  desktopPid,
  pythonExe,
  pythonPath,
  agentRoot,
  uninstallArgs,
  appPath,
  hermesHome
}) {
  const pid = Number(desktopPid) || 0
  // cmd.exe has no string escaping inside quotes; strip embedded quotes (paths
  // under %LOCALAPPDATA% never contain them). `&`/`^` in a path would still be
  // a problem, but Hermes install paths don't use them.
  const q = s => `"${String(s).replace(/"/g, '')}"`

  const lines = [
    '@echo off',
    'setlocal enableextensions',
    `set "HERMES_HOME=${String(hermesHome).replace(/"/g, '')}"`,
    `set "PID=${pid}"`
  ]

  if (pythonPath) {
    lines.push(`set "PYTHONPATH=${String(pythonPath).replace(/"/g, '')};%PYTHONPATH%"`)
  }

  lines.push(
    'set /a waited=0',
    ':waitloop',
    'rem /FI "PID eq %PID%" is an EXACT filter — tasklist outputs the one task',
    'rem row for that PID, or "INFO: No tasks..." otherwise. /NH drops the',
    'rem header; findstr matches the PID as a whole space-delimited token so',
    'rem PID 99 cannot match 990 (the substring trap of a bare `find`).',
    'tasklist /NH /FI "PID eq %PID%" 2>nul | findstr /r /c:" %PID% " >nul',
    'if %ERRORLEVEL% neq 0 goto waited_done',
    'set /a waited+=1',
    'if %waited% geq 60 goto waited_done',
    'timeout /t 1 /nobreak >nul',
    'goto waitloop',
    ':waited_done',
    `cd /d ${q(agentRoot)}`,
    `${q(pythonExe)} ${uninstallArgs.map(q).join(' ')}`
  )

  if (appPath) {
    lines.push(
      'set /a tries=0',
      ':rmloop',
      `if not exist ${q(appPath)} goto rmdone`,
      `rmdir /s /q ${q(appPath)} >nul 2>&1`,
      `if not exist ${q(appPath)} goto rmdone`,
      'set /a tries+=1',
      'if %tries% geq 10 goto rmdone',
      'timeout /t 1 /nobreak >nul',
      'goto rmloop',
      ':rmdone'
    )
  }

  lines.push('del "%~f0"')
  lines.push('')

  return lines.join('\r\n')
}

export {
  allowedUninstallModes,
  buildPosixCleanupScript,
  buildWindowsCleanupScript,
  INSTALL_KINDS,
  installKindAllowsCodeRemoval,
  modeRemovesAgent,
  modeRemovesUserData,
  nativeRemovalInstructions,
  resolveInstallKind,
  resolveRemovableAppPath,
  shouldRemoveAppBundle,
  UNINSTALL_MODES,
  uninstallArgsForMode
}
