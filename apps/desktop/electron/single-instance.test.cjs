'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ELECTRON_DIR = __dirname

function readMainCjs() {
  return fs.readFileSync(path.join(ELECTRON_DIR, 'main.cjs'), 'utf8').replace(/\r\n/g, '\n')
}

function findMatchingBrace(source, openBraceIndex) {
  assert.equal(source[openBraceIndex], '{', 'expected open brace')

  let depth = 0
  for (let index = openBraceIndex; index < source.length; index += 1) {
    if (source[index] === '{') depth += 1
    if (source[index] === '}') depth -= 1
    if (depth === 0) return index
  }

  assert.fail('missing matching close brace')
}

function singleInstanceBranches(source) {
  const ifMarker = 'if (!_gotSingleInstanceLock) {'
  const ifStart = source.indexOf(ifMarker)
  assert.notEqual(ifStart, -1, `missing: ${ifMarker}`)

  const failOpenBrace = source.indexOf('{', ifStart)
  const failCloseBrace = findMatchingBrace(source, failOpenBrace)
  const elseMarker = '} else {'
  assert.equal(source.slice(failCloseBrace, failCloseBrace + elseMarker.length), elseMarker)

  const successOpenBrace = source.indexOf('{', failCloseBrace)
  const successCloseBrace = findMatchingBrace(source, successOpenBrace)

  return {
    failure: source.slice(failOpenBrace + 1, failCloseBrace),
    success: source.slice(successOpenBrace + 1, successCloseBrace),
  }
}

test('single-instance lock failure branch does not reach startup', () => {
  const source = readMainCjs()

  assert.match(source, /app\.requestSingleInstanceLock\(\)/, 'missing requestSingleInstanceLock()')

  const { failure } = singleInstanceBranches(source)
  assert.match(failure, /app\.quit\(\)/, 'lock failure branch must call app.quit()')
  assert.doesNotMatch(failure, /app\.whenReady\(\)/, 'lock failure branch must not register startup')
  assert.doesNotMatch(failure, /app\.on\('open-url'/, 'lock failure branch must not register open-url')
  assert.doesNotMatch(failure, /createWindow\(/, 'lock failure branch must not call createWindow()')
  assert.doesNotMatch(failure, /startHermes\(/, 'lock failure branch must not call startHermes()')
})

test('successful lock path registers second-instance, open-url, and whenReady startup', () => {
  const source = readMainCjs()

  const { success } = singleInstanceBranches(source)

  assert.match(success, /app\.on\('second-instance'/, 'missing second-instance handler')
  assert.match(success, /_extractDeepLink\(argv\)/, 'missing deep-link extraction in second-instance')
  assert.match(success, /handleDeepLink\(url\)/, 'missing handleDeepLink in second-instance')
  assert.match(success, /mainWindow\.focus\(\)/, 'missing window focus in second-instance')

  assert.match(success, /app\.on\('open-url'/, 'missing open-url handler')
  assert.match(success, /event\.preventDefault\(\)/, 'missing preventDefault in open-url')
  assert.match(success, /handleDeepLink\(url\)/, 'missing handleDeepLink in open-url')

  assert.match(success, /app\.whenReady\(\)\.then\(/, 'missing whenReady startup')
  assert.match(success, /createWindow\(\)/, 'missing createWindow() in startup')
})

test('before-quit and window-all-closed cleanup remain registered', () => {
  const source = readMainCjs()

  assert.match(source, /app\.on\('before-quit'/, 'missing before-quit handler')
  assert.match(source, /app\.on\('window-all-closed'/, 'missing window-all-closed handler')
})

test('configureSpellChecker is defined', () => {
  const source = readMainCjs()

  assert.match(source, /function configureSpellChecker\(\)/, 'missing configureSpellChecker definition')
})
