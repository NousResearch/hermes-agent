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
 * Map an uninstall mode to the `python -m hermes_cli.uninstall` argv (after the
 * python executable). Uses the dedicated lightweight module entrypoint (not
 * `hermes_cli.main`) so it can run under a system Python OUTSIDE the venv that
 * lite/full delete — see the Finding-3 note in buildWindowsCleanupScript.
 * Throws on an unknown mode so a typo can't silently become a full wipe.
 */
function uninstallArgsForMode(mode) {
  if (!UNINSTALL_MODES.includes(mode)) {
    throw new Error(`Unknown uninstall mode: ${mode}`)
  }
  return ['-m', 'hermes_cli.uninstall', '--mode', mode]
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
    '#!/bin/bash',
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
