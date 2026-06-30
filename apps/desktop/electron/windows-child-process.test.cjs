'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ELECTRON_DIR = __dirname

function readElectronFile(name) {
  return fs.readFileSync(path.join(ELECTRON_DIR, name), 'utf8').replace(/\r\n/g, '\n')
}

function snippetFor(source, needle) {
  const match = needle instanceof RegExp ? needle.exec(source) : null
  const index = needle instanceof RegExp ? (match?.index ?? -1) : source.indexOf(needle)
  assert.notEqual(index, -1, `missing call site: ${needle}`)
  return source.slice(index, index + 700)
}

function requireHiddenChildOptions(source, needle) {
  assert.match(
    snippetFor(source, needle),
    /hiddenWindowsChildOptions\(/,
    `expected ${needle} to wrap child-process options with hiddenWindowsChildOptions`
  )
}

test('remote-only desktop has no local backend spawn command sites', () => {
  const source = readElectronFile('main.cjs')

  assert.doesNotMatch(source, /hermesProcess = spawn\(/)
  assert.doesNotMatch(source, /spawn\(\s*backend\.command,\s*backend\.args/)
  assert.doesNotMatch(source, /'-m', 'hermes_cli\.main'/)
  assert.doesNotMatch(source, /runBootstrap\(/)
  assert.doesNotMatch(source, /spawn\(updater/)
})

test('remaining background child processes opt into hidden Windows consoles', () => {
  const source = readElectronFile('main.cjs')

  assert.match(source, /function hiddenWindowsChildOptions\(options = \{\}\)/)
  requireHiddenChildOptions(source, "execFileSync('taskkill'")
  requireHiddenChildOptions(source, "spawn('curl'")
  assert.match(snippetFor(source, /spawn\(\s*runner,\s*runnerArgs/), /windowsHide:\s*true/)
})

test('intentional interactive child processes stay documented', () => {
  const source = readElectronFile('main.cjs')

  assert.match(source, /nodePty\.spawn\(command, args/)
  assert.match(source, /spawn\('cmd\.exe', \['\/c', 'start'/)
})
