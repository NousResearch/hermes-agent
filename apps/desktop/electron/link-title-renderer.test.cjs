'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ELECTRON_DIR = __dirname

function readElectronFile(name) {
  return fs.readFileSync(path.join(ELECTRON_DIR, name), 'utf8').replace(/\r\n/g, '\n')
}

test('renderer link-title timeouts tolerate destroyed BrowserWindow objects', () => {
  const source = readElectronFile('main.cjs')
  const readTitleIndex = source.indexOf('const readTitle = () => {')
  assert.notEqual(readTitleIndex, -1, 'missing renderer title reader')

  const snippet = source.slice(readTitleIndex, readTitleIndex + 900)
  assert.match(snippet, /try \{/, 'title reader should guard Electron destroyed-object races')
  assert.match(snippet, /window\.isDestroyed\(\)/, 'BrowserWindow must be checked before webContents access')
  assert.match(snippet, /webContents\.isDestroyed\(\)/, 'WebContents must be checked before getTitle')
  assert.match(snippet, /catch \{[\s\S]*return ''[\s\S]*\}/, 'destroyed-object TypeError should resolve to no title')
  assert.match(snippet, /webContents\.getTitle\?\.\(\)/, 'title reader should still return the renderer title when live')
})
