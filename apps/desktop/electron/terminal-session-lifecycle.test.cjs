'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ELECTRON_DIR = __dirname

function readElectronFile(name) {
  return fs.readFileSync(path.join(ELECTRON_DIR, name), 'utf8').replace(/\r\n/g, '\n')
}

test('embedded terminal sessions dispose node-pty listeners before app quit', () => {
  const source = readElectronFile('main.cjs')

  assert.match(source, /function disposeAllTerminalSessions\(\)/)
  assert.match(source, /function disposeTerminalSessionListeners\(sessionInfo\)/)
  assert.match(source, /const dataDisposable = ptyProcess\.onData/)
  assert.match(source, /const exitDisposable = ptyProcess\.onExit/)
  assert.match(source, /disposables: \[\], removeSenderDestroyedListener: null/)
  assert.match(source, /sessionInfo\.disposables\.push\(dataDisposable, exitDisposable\)/)
  assert.match(source, /removeSenderDestroyedListener/)
  assert.match(source, /event\.sender\.off\('destroyed', senderDestroyedListener\)/)

  const beforeQuit = source.slice(source.indexOf("app.on('before-quit'"), source.indexOf("app.on('window-all-closed'"))
  assert.match(beforeQuit, /disposeAllTerminalSessions\(\)/)
})
