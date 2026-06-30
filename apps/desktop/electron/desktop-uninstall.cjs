'use strict'

/**
 * Pure helpers for the Reuben desktop app uninstaller.
 *
 * Reuben Desktop is a strict remote frontend, so uninstall never shells into
 * hermes_cli and never removes backend code, backend config, or ~/.hermes.
 * The only supported cleanup is the packaged desktop app bundle itself.
 */

const path = require('node:path')

const UNINSTALL_MODES = ['app']

function assertAppUninstallMode(mode) {
  if (!UNINSTALL_MODES.includes(mode)) {
    throw new Error(`Unknown uninstall mode: ${mode}`)
  }
}

/**
 * Resolve the on-disk app bundle/dir to remove for the running desktop app,
 * given the path to the running executable (`process.execPath`) and platform.
 *
 *   macOS:   …/Reuben.app/Contents/MacOS/Reuben → …/Reuben.app
 *   Windows: …\Reuben\Reuben.exe                → …\Reuben
 *   Linux:   AppImage → the APPIMAGE env path; unpacked → the *-unpacked dir
 */
function resolveRemovableAppPath(execPath, platform, env = {}) {
  const exe = String(execPath || '')
  if (!exe) return null

  const p = platform === 'win32' ? path.win32 : path.posix

  if (platform === 'darwin') {
    const macOsDir = p.dirname(exe)
    const contents = p.dirname(macOsDir)
    const appBundle = p.dirname(contents)
    return appBundle.endsWith('.app') ? appBundle : null
  }

  if (platform === 'win32') {
    const dir = p.dirname(exe)
    return /[\\/]Reuben$/i.test(dir) || /[\\/]reuben-desktop$/i.test(dir) ? dir : null
  }

  if (env.APPIMAGE) return env.APPIMAGE
  const dir = p.dirname(exe)
  return /-unpacked$/.test(dir) ? dir : null
}

function shouldRemoveAppBundle(isPackaged, appPath) {
  return Boolean(isPackaged) && Boolean(appPath)
}

function buildAppOnlyCleanupScript({ desktopPid, appPath, platform }) {
  if (platform === 'win32') {
    const pid = Number(desktopPid) || 0
    const q = s => `"${String(s).replace(/"/g, '')}"`
    return [
      '@echo off',
      'setlocal enableextensions',
      `set "PID=${pid}"`,
      'set /a waited=0',
      ':waitloop',
      'tasklist /NH /FI "PID eq %PID%" 2>nul | findstr /r /c:" %PID% " >nul',
      'if %ERRORLEVEL% neq 0 goto waited_done',
      'set /a waited+=1',
      'if %waited% geq 60 goto waited_done',
      'timeout /t 1 /nobreak >nul',
      'goto waitloop',
      ':waited_done',
      'set /a tries=0',
      ':rmloop',
      `if not exist ${q(appPath)} goto rmdone`,
      `rmdir /s /q ${q(appPath)} >nul 2>&1`,
      `if not exist ${q(appPath)} goto rmdone`,
      'set /a tries+=1',
      'if %tries% geq 10 goto rmdone',
      'timeout /t 1 /nobreak >nul',
      'goto rmloop',
      ':rmdone',
      'del "%~f0"',
      ''
    ].join('\r\n')
  }

  const q = s => `'${String(s).replace(/'/g, `'\\''`)}'`
  return [
    '#!/bin/bash',
    'set -u',
    `pid=${Number(desktopPid) || 0}`,
    'if [ "$pid" -gt 0 ]; then',
    '  for _ in $(seq 1 60); do',
    '    kill -0 "$pid" 2>/dev/null || break',
    '    sleep 0.5',
    '  done',
    'fi',
    `rm -rf ${q(appPath)} || true`,
    'rm -f "$0" 2>/dev/null || true',
    ''
  ].join('\n')
}

module.exports = {
  UNINSTALL_MODES,
  assertAppUninstallMode,
  buildAppOnlyCleanupScript,
  resolveRemovableAppPath,
  shouldRemoveAppBundle
}
