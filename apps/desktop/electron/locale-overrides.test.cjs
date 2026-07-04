'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const { readLocaleOverridesForIpc, localeOverridesFile } = require('./locale-overrides.cjs')

function mkHome() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-locale-ov-'))
}

function writeOverride(home, lang, obj) {
  const dir = path.join(home, 'locale-overrides', 'desktop')
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, `${lang}.json`), JSON.stringify(obj), 'utf8')
}

test('reads a valid override object', () => {
  const home = mkHome()
  writeOverride(home, 'ja', { common: { save: '保存する' } })
  assert.deepEqual(readLocaleOverridesForIpc(home, 'ja'), { common: { save: '保存する' } })
})

test('returns null when no override file exists', () => {
  const home = mkHome()
  assert.equal(readLocaleOverridesForIpc(home, 'ja'), null)
})

test('returns null for malformed JSON', () => {
  const home = mkHome()
  const dir = path.join(home, 'locale-overrides', 'desktop')
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, 'ja.json'), '{ not json', 'utf8')
  assert.equal(readLocaleOverridesForIpc(home, 'ja'), null)
})

test('returns null for a JSON array or scalar (must be an object)', () => {
  const home = mkHome()
  const dir = path.join(home, 'locale-overrides', 'desktop')
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, 'ja.json'), '["a","b"]', 'utf8')
  assert.equal(readLocaleOverridesForIpc(home, 'ja'), null)
})

test('rejects path-traversal and bogus language tokens', () => {
  const home = mkHome()
  assert.equal(localeOverridesFile(home, '../../etc/passwd'), null)
  assert.equal(localeOverridesFile(home, 'ja/../../secret'), null)
  assert.equal(localeOverridesFile(home, ''), null)
  assert.equal(localeOverridesFile(home, '.'), null)
  assert.equal(readLocaleOverridesForIpc(home, '../../etc/passwd'), null)
})

test('rejects an empty hermes home', () => {
  assert.equal(localeOverridesFile('', 'ja'), null)
  assert.equal(readLocaleOverridesForIpc('', 'ja'), null)
})

test('builds the expected path for a valid locale', () => {
  const home = mkHome()
  const file = localeOverridesFile(home, 'zh-hant')
  assert.equal(file, path.join(home, 'locale-overrides', 'desktop', 'zh-hant.json'))
})
