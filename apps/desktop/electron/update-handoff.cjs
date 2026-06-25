const fs = require('node:fs')
const path = require('node:path')

function quoteBatchArg(value) {
  const escaped = String(value)
    .replace(/%/g, '%%')
    .replace(/"/g, '""')
  return `"${escaped}"`
}

function buildHermesUpdateArgs({ assumeYes = false, branch = null } = {}) {
  const args = ['update']
  if (assumeYes) args.push('--yes')
  args.push('--backup')
  if (branch) args.push('--branch', branch)
  return args
}

function buildManualHermesUpdateCommand(branch = null) {
  return ['hermes', ...buildHermesUpdateArgs({ branch })].join(' ')
}

function buildVisibleWindowsUpdaterScript(updater, updaterArgs = []) {
  const command = [quoteBatchArg(updater), ...updaterArgs.map(quoteBatchArg)].join(' ')
  return [
    '@echo off',
    'setlocal',
    'title Hermes update',
    'echo [Hermes] Closing Hermes runtime processes before update...',
    'echo [Hermes] Running Hermes updater...',
    'echo [Hermes] Relaunching Hermes when the updater finishes...',
    'echo.',
    `echo [Hermes] Command: ${command}`,
    command,
    'set "HERMES_UPDATE_EXIT=%ERRORLEVEL%"',
    'if "%HERMES_UPDATE_EXIT%"=="0" (',
    '  echo.',
    '  echo [Hermes] Update command finished. Closing this window shortly...',
    '  timeout /t 2 /nobreak >nul',
    ')',
    'if not "%HERMES_UPDATE_EXIT%"=="0" (',
    '  echo.',
    '  echo [Hermes] Updater failed with exit code %HERMES_UPDATE_EXIT%.',
    '  echo [Hermes] Press any key to close this window.',
    '  pause >nul',
    ')',
    'exit /b %HERMES_UPDATE_EXIT%',
    ''
  ].join('\r\n')
}

function createUpdaterLaunchPlan({ handoffDir, isWindows = process.platform === 'win32', updater, updaterArgs = [] }) {
  if (!isWindows) {
    return {
      args: updaterArgs,
      command: updater,
      detached: true,
      scriptPath: null,
      windowsHide: false
    }
  }

  if (!handoffDir) {
    throw new Error('createUpdaterLaunchPlan requires handoffDir on Windows')
  }

  fs.mkdirSync(handoffDir, { recursive: true })
  const scriptPath = path.join(handoffDir, `hermes-updater-${process.pid}-${Date.now()}.cmd`)
  fs.writeFileSync(scriptPath, buildVisibleWindowsUpdaterScript(updater, updaterArgs), 'utf8')

  return {
    args: ['/d', '/s', '/c', scriptPath],
    command: 'cmd.exe',
    detached: true,
    scriptPath,
    windowsHide: false
  }
}

module.exports = {
  buildHermesUpdateArgs,
  buildManualHermesUpdateCommand,
  buildVisibleWindowsUpdaterScript,
  createUpdaterLaunchPlan
}
