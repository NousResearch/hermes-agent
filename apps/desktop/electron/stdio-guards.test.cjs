'use strict'

const assert = require('node:assert/strict')
const { EventEmitter } = require('node:events')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

const { installStdioPipeErrorGuards, isIgnorablePipeError } = require('./stdio-guards.cjs')

function codedError(code, message = code) {
  const error = new Error(message)
  error.code = code
  return error
}

test('stdio pipe guard classifies closed-pipe write errors as ignorable', () => {
  assert.equal(isIgnorablePipeError(codedError('EPIPE', 'broken pipe, write')), true)
  assert.equal(isIgnorablePipeError(codedError('ERR_STREAM_DESTROYED', 'Cannot call write after destroy')), true)
  assert.equal(isIgnorablePipeError(new Error('write EPIPE')), true)
  assert.equal(isIgnorablePipeError(codedError('EACCES', 'permission denied')), false)
  assert.equal(isIgnorablePipeError(null), false)
})

test('stdio pipe guard suppresses EPIPE from closed inherited stdio', () => {
  const stdout = new EventEmitter()
  const stderr = new EventEmitter()

  assert.equal(installStdioPipeErrorGuards({ stdout, stderr }), 2)
  assert.doesNotThrow(() => stderr.emit('error', codedError('EPIPE', 'broken pipe, write')))
  assert.doesNotThrow(() => stdout.emit('error', codedError('ERR_STREAM_DESTROYED')))
})

test('stdio pipe guard preserves non-pipe stream errors', () => {
  const stderr = new EventEmitter()

  installStdioPipeErrorGuards({ stdout: null, stderr })

  assert.throws(() => stderr.emit('error', codedError('EACCES', 'permission denied')), /permission denied/)
})

test('stdio pipe guard installs at most once per stream', () => {
  const stdout = new EventEmitter()

  assert.equal(installStdioPipeErrorGuards({ stdout, stderr: null }), 1)
  assert.equal(installStdioPipeErrorGuards({ stdout, stderr: null }), 0)
  assert.equal(stdout.listenerCount('error'), 1)
})

test('Electron main installs stdio guards before loading electron', () => {
  const source = fs.readFileSync(path.join(__dirname, 'main.cjs'), 'utf8')
  const guardImport = source.indexOf("require('./stdio-guards.cjs')")
  const guardInstall = source.indexOf('installStdioPipeErrorGuards()')
  const electronImport = source.indexOf("require('electron')")

  assert.notEqual(guardImport, -1)
  assert.notEqual(guardInstall, -1)
  assert.notEqual(electronImport, -1)
  assert.ok(guardImport < electronImport, 'stdio guard module must load before electron')
  assert.ok(guardInstall < electronImport, 'stdio guards must install before electron initializes')
})

test('WSL external links avoid cmd.exe command-shell URL parsing', () => {
  const source = fs.readFileSync(path.join(__dirname, 'main.cjs'), 'utf8')
  const opener = source.slice(source.indexOf('if (IS_WSL)'), source.indexOf('shell.openExternal(url)'))

  assert.match(opener, /spawn\('powershell\.exe'/)
  assert.match(opener, /Start-Process -FilePath \$args\[0\]/)
  assert.doesNotMatch(opener, /cmd\.exe/)
  assert.doesNotMatch(opener, /'\/c'/)
})
