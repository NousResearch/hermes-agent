'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')
const { pathToFileURL } = require('node:url')

const { gitRootForIpc } = require('./git-root.cjs')

function mkTmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-git-root-'))
}

test('gitRootForIpc returns null for invalid and device paths', async () => {
  assert.equal(await gitRootForIpc(''), null)
  assert.equal(await gitRootForIpc('   '), null)
  assert.equal(await gitRootForIpc(null), null)
  assert.equal(await gitRootForIpc('\\\\?\\C:\\secret'), null)
  assert.equal(await gitRootForIpc('file:///%E0%A4%A'), null)
})

test('gitRootForIpc resolves directories files missing descendants and file URLs', async t => {
  const root = mkTmpDir()
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))

  const gitDir = path.join(root, '.git')
  const srcDir = path.join(root, 'src')
  const filePath = path.join(srcDir, 'index.ts')
  fs.mkdirSync(gitDir)
  fs.mkdirSync(srcDir)
  fs.writeFileSync(filePath, 'export {}\n', 'utf8')

  assert.equal(await gitRootForIpc(root), root)
  assert.equal(await gitRootForIpc(srcDir), root)
  assert.equal(await gitRootForIpc(filePath), root)
  assert.equal(await gitRootForIpc(pathToFileURL(filePath).toString()), root)
  assert.equal(await gitRootForIpc(path.join(srcDir, 'missing.ts')), root)
})
