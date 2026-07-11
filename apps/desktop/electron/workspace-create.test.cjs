'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const { createProjectDirForIpc, validateProjectName } = require('./workspace-create.cjs')

function mkTmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-workspace-create-'))
}

test('validateProjectName accepts real project names', () => {
  assert.equal(validateProjectName('hermes-agent'), 'hermes-agent')
  assert.equal(validateProjectName('  My Project  '), 'My Project')
  assert.equal(validateProjectName('app_v2.1'), 'app_v2.1')
})

test('validateProjectName rejects empty, traversal, and separators', () => {
  for (const bad of ['', '   ', '.', '..', 'a/b', 'a\\b', '../escape']) {
    assert.throws(() => validateProjectName(bad), /Project name/, `expected reject: ${JSON.stringify(bad)}`)
  }
})

test('createProjectDirForIpc creates a child folder and returns its path', async t => {
  const parent = mkTmpDir()
  t.after(() => fs.rmSync(parent, { recursive: true, force: true }))

  const result = await createProjectDirForIpc(parent, 'my-new-project')

  assert.equal(result.ok, true)
  assert.equal(result.path, path.join(parent, 'my-new-project'))
  assert.equal(fs.statSync(result.path).isDirectory(), true)
})

test('createProjectDirForIpc refuses to clobber an existing entry', async t => {
  const parent = mkTmpDir()
  t.after(() => fs.rmSync(parent, { recursive: true, force: true }))
  fs.mkdirSync(path.join(parent, 'taken'))

  const result = await createProjectDirForIpc(parent, 'taken')

  assert.equal(result.ok, false)
  assert.equal(result.code, 'EEXIST')
  assert.equal(result.path, null)
})

test('createProjectDirForIpc rejects an invalid name without touching disk', async t => {
  const parent = mkTmpDir()
  t.after(() => fs.rmSync(parent, { recursive: true, force: true }))

  const result = await createProjectDirForIpc(parent, '../escape')

  assert.equal(result.ok, false)
  assert.equal(result.code, 'invalid-name')
  // Nothing was created outside the parent.
  assert.equal(fs.existsSync(path.join(path.dirname(parent), 'escape')), false)
})

test('createProjectDirForIpc fails when the parent does not exist', async () => {
  const result = await createProjectDirForIpc(path.join(os.tmpdir(), 'hermes-nope-does-not-exist-xyz'), 'proj')

  assert.equal(result.ok, false)
  assert.equal(result.code, 'ENOENT')
})

test('createProjectDirForIpc fails when the parent is a file', async t => {
  const dir = mkTmpDir()
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const filePath = path.join(dir, 'a-file')
  fs.writeFileSync(filePath, 'x', 'utf8')

  const result = await createProjectDirForIpc(filePath, 'proj')

  assert.equal(result.ok, false)
  assert.equal(result.code, 'ENOTDIR')
})

test('createProjectDirForIpc rejects blank / invalid parent paths', async () => {
  const blank = await createProjectDirForIpc('   ', 'proj')
  assert.equal(blank.ok, false)
  assert.equal(blank.code, 'invalid-path')
})
