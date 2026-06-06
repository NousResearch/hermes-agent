/**
 * desktop-uninstall.cjs
 *
 * Pure, electron-free helpers for the desktop Chat GUI uninstaller. These map
 * the three user-facing uninstall modes to the `hermes uninstall` CLI flags,
 * resolve the running app bundle/exe so a detached cleanup script can remove
 * it after the app quits, and build that cleanup script for each OS.
 *
 * Kept standalone (no `require('electron')`) so it can be unit-tested with
 * `node --test` — same pattern as connection-config.cjs / backend-probes.cjs.
 * main.cjs requires these and wires them into the electron-coupled IPC layer.
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
 * shape as the self-update swap-and-relaunch flow already in main.cjs.
 */

const path = require('node:path')

const UNINSTALL_MODES = ['gui', 'lite', 'full']

/**
 * Map an uninstall mode to the `hermes_cli.main uninstall` argv (after the
 * `-m hermes_cli.main` prefix). Always non-interactive (`--yes`).
 * Throws on an unknown mode so a typo can't silently become a full wipe.
 */
function uninstallArgsForMode(mode) {
  switch (mode) {
    case 'gui':
      return ['uninstall', '--gui', '--yes']
    case 'lite':
      return ['uninstall', '--yes']
    case 'full':
      return ['uninstall', '--full', '--yes']
    default:
      throw new Error(`Unknown uninstall mode: ${mode}`)
  }
}

/** True when `mode` removes the agent (lite/full), false for gui-only. */
function modeRemovesAgent(mode) {
  return mode === 'lite' || mode === 'full'
}

/** True when `mode` removes user data (full only). */
function modeRemovesUserData(mode) {
  return mode === 'full'
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
function resolveRemovableAppPath(execPath, platform, env = {}) {
  const exe = String(execPath || '')
  if (!exe) return null

  // Use the path flavor that matches the TARGET platform, not the host running
  // this code — so the Windows branch parses backslash paths correctly even
  // when these pure helpers are unit-tested on Linux/macOS CI.
  const p = platform === 'win32' ? path.win32 : path.posix

  if (platform === 'darwin') {
    // …/Hermes.app/Contents/MacOS/Hermes → strip 3 segments to the .app
    const macOsDir = p.dirname(exe) // …/Contents/MacOS
    const contents = p.dirname(macOsDir) // …/Contents
    const appBundle = p.dirname(contents) // …/Hermes.app
    if (appBundle.endsWith('.app')) return appBundle
    return null
  }

  if (platform === 'win32') {
    // NSIS per-user installs Hermes.exe directly in the install dir.
    const dir = p.dirname(exe)
    if (/[\\/]Hermes$/i.test(dir) || /[\\/]hermes-desktop$/i.test(dir)) return dir
    return null
  }

  // Linux: an AppImage exposes its own path via the APPIMAGE env var.
  if (env.APPIMAGE) return env.APPIMAGE
  // Unpacked electron-builder tree: …/linux-unpacked/hermes
  const dir = p.dirname(exe)
  if (/-unpacked$/.test(dir)) return dir
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
 *   1. waits for the desktop PID to exit (so the venv shim + bundle unlock),
 *   2. runs the Python uninstall with the mode's flags,
 *   3. removes the app bundle if one was resolved.
 * `quote` defends against spaces in paths.
 */
function buildPosixCleanupScript({ desktopPid, pythonExe, agentRoot, uninstallArgs, appPath, hermesHome }) {
  const q = s => `'${String(s).replace(/'/g, `'\\''`)}'`
  const lines = [
    '#!/bin/bash',
    'set -u',
    '# Wait (up to ~30s) for the desktop process to exit so the venv python',
    '# shim and the app bundle are no longer locked/in-use.',
    `pid=${Number(desktopPid) || 0}`,
    'if [ "$pid" -gt 0 ]; then',
    '  for _ in $(seq 1 60); do',
    '    kill -0 "$pid" 2>/dev/null || break',
    '    sleep 0.5',
    '  done',
    'fi',
    `export HERMES_HOME=${q(hermesHome)}`,
    `cd ${q(agentRoot)} 2>/dev/null || true`,
    `${q(pythonExe)} -m hermes_cli.main ${uninstallArgs.map(q).join(' ')} || true`
  ]
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
 * Uses tasklist to wait for the desktop PID to exit, then runs the uninstall
 * and removes the install dir.
 *
 * Windows-specific hardening: even after the desktop PID is gone, a freshly
 * SIGTERM'd backend grandchild can briefly keep a handle open, and the OS
 * releases directory handles lazily — so a single `rmdir /s /q` can half-fail
 * with the tree still partly present. The Electron side already tree-kills all
 * backends + waits for the venv shim to unlock before launching this script
 * (releaseBackendLock), but we belt-and-suspenders it here: tree-kill anything
 * still rooted under the install dir, then retry the rmdir a few times with a
 * short pause so a lazily-released handle can't leave a half-removed bundle.
 */
function buildWindowsCleanupScript({ desktopPid, pythonExe, agentRoot, uninstallArgs, appPath, hermesHome }) {
  const pid = Number(desktopPid) || 0
  const q = s => `"${String(s).replace(/"/g, '')}"`
  const lines = [
    '@echo off',
    'setlocal enableextensions',
    `set HERMES_HOME=${hermesHome}`,
    `set PID=${pid}`,
    ':waitloop',
    'tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul',
    'if %ERRORLEVEL%==0 (',
    '  timeout /t 1 /nobreak >nul',
    '  goto waitloop',
    ')',
    `cd /d ${q(agentRoot)}`,
    `${q(pythonExe)} -m hermes_cli.main ${uninstallArgs.map(q).join(' ')}`
  ]
  if (appPath) {
    // The Electron side already tree-killed every backend + waited for the
    // venv shim to unlock before launching this script (releaseBackendLock),
    // so handles should be gone. But Windows releases directory handles
    // lazily, so a single rmdir can still half-fail — retry up to 10x with a
    // 1s pause until the tree is actually gone.
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

module.exports = {
  UNINSTALL_MODES,
  buildPosixCleanupScript,
  buildWindowsCleanupScript,
  modeRemovesAgent,
  modeRemovesUserData,
  resolveRemovableAppPath,
  shouldRemoveAppBundle,
  uninstallArgsForMode
}
