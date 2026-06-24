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
  const index = source.indexOf(needle)
  assert.notEqual(index, -1, `missing call site: ${needle}`)
  const snippet = source.slice(index, index + 700)
  assert.match(
    snippet,
    /hiddenWindowsChildOptions\(/,
    `expected ${needle} to wrap child-process options with hiddenWindowsChildOptions`
  )
}

function requireHiddenChildOptionsNear(source, pattern, label) {
  const match = source.match(pattern)
  assert.ok(match && typeof match.index === 'number', `missing call site: ${label}`)
  const snippet = source.slice(match.index, match.index + 900)
  assert.match(
    snippet,
    /hiddenWindowsChildOptions\(/,
    `expected ${label} to wrap child-process options with hiddenWindowsChildOptions`
  )
}

test('desktop background child processes opt into hidden Windows consoles', () => {
  const source = readElectronFile('main.cjs')

  assert.match(source, /function hiddenWindowsChildOptions\(options = \{\}\)/)

  requireHiddenChildOptions(source, "execFileSync(\n          'reg'")
  requireHiddenChildOptionsNear(source, /execFileSync\(\s*pyExe/, 'execFileSync(pyExe')
  requireHiddenChildOptionsNear(source, /spawn\(\s*resolveGitBinary\(\)/, 'spawn(resolveGitBinary()')
  requireHiddenChildOptions(source, "execFileSync('taskkill'")
  requireHiddenChildOptions(source, "spawn('curl'")
  requireHiddenChildOptionsNear(source, /const child = spawn\(\s*backend\.command,\s*backend\.args/, 'spawn(backend.command, backend.args')
  requireHiddenChildOptionsNear(source, /hermesProcess = spawn\(\s*backend\.command,\s*backend\.args/, 'hermesProcess = spawn(backend.command, backend.args')
  requireHiddenChildOptionsNear(source, /spawn\(\s*py,\s*\[\s*'-m',\s*'hermes_cli\.main',\s*'uninstall',\s*'--gui-summary'\s*\]/, "spawn(py, ['-m', 'hermes_cli.main', 'uninstall', '--gui-summary']")
})

test('intentional or interactive desktop child processes stay documented', () => {
  const source = readElectronFile('main.cjs')

  assert.match(source, /windowsHide: false/)
  assert.match(source, /handOffWindowsBootstrapRecovery/)
  assert.match(source, /'--repair', '--branch'/)
  assert.match(source, /'--update', '--branch'/)
  assert.match(source, /nodePty\.spawn\(command, args/)
  assert.match(source, /spawn\('cmd\.exe', \['\/c', 'start'/)
})

test('bootstrap PowerShell runner hides Windows console children', () => {
  const source = readElectronFile('bootstrap-runner.cjs')

  assert.match(source, /function hiddenWindowsChildOptions\(options = \{\}\)/)
  requireHiddenChildOptions(source, 'spawn(ps, fullArgs')
})
