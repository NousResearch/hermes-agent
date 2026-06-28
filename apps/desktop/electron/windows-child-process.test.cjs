'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ELECTRON_DIR = __dirname

function readElectronFile(name) {
  return fs.readFileSync(path.join(ELECTRON_DIR, name), 'utf8').replace(/\r\n/g, '\n')
}

function requireHiddenChildOptions(source, needle) {
  const match = needle instanceof RegExp ? needle.exec(source) : null
  const index = needle instanceof RegExp ? (match?.index ?? -1) : source.indexOf(needle)
  assert.notEqual(index, -1, `missing call site: ${needle}`)
  const snippet = source.slice(index, index + 700)
  assert.match(
    snippet,
    /hiddenWindowsChildOptions\(/,
    `expected ${needle} to wrap child-process options with hiddenWindowsChildOptions`
  )
}

test('desktop background child processes opt into hidden Windows consoles', () => {
  const source = readElectronFile('main.cjs')

  assert.match(source, /function hiddenWindowsChildOptions\(options = \{\}\)/)

  requireHiddenChildOptions(source, "execFileSync(\n          'reg'")
  requireHiddenChildOptions(source, /execFileSync\(\s*pyExe/)
  requireHiddenChildOptions(source, /spawn\(\s*resolveGitBinary\(\)/)
  requireHiddenChildOptions(source, "execFileSync('taskkill'")
  requireHiddenChildOptions(source, /spawn\(\s*command,\s*args/)
  requireHiddenChildOptions(source, "spawn('curl'")
  requireHiddenChildOptions(source, /spawn\(\s*backend\.command,\s*backend\.args/)
  requireHiddenChildOptions(source, /hermesProcess = spawn\(\s*backend\.command,\s*backend\.args/)
  requireHiddenChildOptions(source, /spawn\(\s*py,\s*\['-m', 'hermes_cli\.main', 'uninstall', '--gui-summary'\]/)

  assert.match(source, /function unwrapWindowsVenvHermesCommand\(command, dashboardArgs\)/)
  assert.match(source, /existing Hermes no-console Python at/)
  assert.match(source, /function getNoConsoleVenvPython\(venvRoot\)/)
  assert.match(source, /function toNoConsolePython\(pythonPath\)/)
  assert.match(source, /function applyWindowsNoConsoleSpawnHints\(backend\)/)
  assert.match(source, /function readVenvHome\(venvRoot\)/)
  assert.match(source, /path\.join\(venvRoot, 'Scripts', 'pythonw\.exe'\)/)
  assert.match(source, /backendStartFailure/)
  assert.match(source, /HERMES_DESKTOP_READY_FILE/)
  assert.match(source, /readyFile: true/)
  assert.match(source, /function getVenvSitePackagesEntries\(venvRoot\)/)
  assert.match(source, /path\.join\(venvRoot, 'Lib', 'site-packages'\)/)
  assert.match(source, /args: \['-m', 'hermes_cli\.main', \.\.\.dashboardArgs\]/)
})

test('intentional or interactive desktop child processes stay documented', () => {
  const source = readElectronFile('main.cjs')

  assert.match(source, /windowsHide: false/)
  assert.match(source, /handOffWindowsBootstrapRecovery/)
  assert.match(source, /'--repair', '--branch'/)
  assert.match(source, /'--update', '--branch'/)
  assert.match(source, /nodePty\.spawn\(command, args/)
  assert.match(source, /spawn\('powershell\.exe', \['-NoProfile', '-NonInteractive', '-Command', 'Start-Process -FilePath \$args\[0\]'/)
})

test('bootstrap PowerShell runner hides Windows console children', () => {
  const source = readElectronFile('bootstrap-runner.cjs')

  assert.match(source, /function hiddenWindowsChildOptions\(options = \{\}\)/)
  requireHiddenChildOptions(source, 'spawn(ps, fullArgs')
})
