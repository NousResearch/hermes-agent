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

test('desktop background child processes opt into hidden Windows consoles', () => {
  const source = readElectronFile('main.cjs')

  assert.match(source, /function hiddenWindowsChildOptions\(options = \{\}\)/)

  requireHiddenChildOptions(source, "execFileSync(\n          'reg'")
  requireHiddenChildOptions(source, 'execFileSync(\n          pyExe')
  requireHiddenChildOptions(source, 'spawn(\n      resolveGitBinary()')
  requireHiddenChildOptions(source, "execFileSync('taskkill'")
  requireHiddenChildOptions(source, 'spawn(\n        command')
  requireHiddenChildOptions(source, "spawn('curl'")
  requireHiddenChildOptions(source, 'spawn(\n    backend.command')
  requireHiddenChildOptions(source, 'hermesProcess = spawn(\n      backend.command')
  requireHiddenChildOptions(source, "spawn(\n        py,\n        ['-m', 'hermes_cli.main', 'uninstall', '--gui-summary']")
})

test('Windows before-quit waits for backend process trees before exiting', () => {
  const source = readElectronFile('main.cjs')
  const index = source.indexOf("app.on('before-quit', event =>")
  assert.notEqual(index, -1, 'missing before-quit lifecycle handler')

  const helperIndex = source.indexOf('function forceKillWindowsBackendTrees(children)')
  assert.notEqual(helperIndex, -1, 'missing Windows backend tree-kill helper')
  const helper = source.slice(helperIndex, helperIndex + 500)
  assert.match(helper, /isBackendChildRunning\(child\)/)
  assert.match(helper, /forceKillProcessTree\(child\.pid\)/)

  const snippet = source.slice(index, index + 1400)
  assert.match(snippet, /collectBackendChildrenForQuit\(\)/)
  assert.match(snippet, /event\.preventDefault\(\)/)
  assert.match(snippet, /forceKillWindowsBackendTrees\(backendChildren\)/)
  assert.match(snippet, /waitForBackendExit\(child\)/)
  assert.ok(
    snippet.indexOf('forceKillWindowsBackendTrees(backendChildren)') < snippet.indexOf('waitForBackendExit(child)'),
    'backend trees must be killed while parent PIDs still identify descendants'
  )
  assert.match(snippet, /quitBackendTeardownComplete = true/)
  assert.match(snippet, /app\.quit\(\)/)
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
