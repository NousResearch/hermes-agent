const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const { createFolderForIpc, createTextFileForIpc } = require('./fs-create.cjs')

function mkTmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-fs-create-'))
}

test('createTextFileForIpc creates a new UTF-8 file without overwriting', async t => {
  const root = mkTmpDir()
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))

  const target = path.join(root, 'TODO.md')
  const result = await createTextFileForIpc(target, '# Todo\n')

  assert.equal(result.path, target)
  assert.equal(fs.readFileSync(target, 'utf8'), '# Todo\n')

  await assert.rejects(() => createTextFileForIpc(target, 'replace'), { code: 'EEXIST' })
})

test('createFolderForIpc creates exactly one new folder without overwriting', async t => {
  const root = mkTmpDir()
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))

  const target = path.join(root, 'docs')
  const result = await createFolderForIpc(target)

  assert.equal(result.path, target)
  assert.equal(fs.statSync(target).isDirectory(), true)

  await assert.rejects(() => createFolderForIpc(target), { code: 'EEXIST' })
})

test('create helpers reject invalid device paths and missing parents', async () => {
  await assert.rejects(() => createTextFileForIpc(''), /path is required/)
  await assert.rejects(() => createFolderForIpc('\\\\?\\C:\\secret'), { code: 'device-path' })

  const root = mkTmpDir()
  try {
    await assert.rejects(() => createTextFileForIpc(path.join(root, 'missing', 'file.txt')), { code: 'ENOENT' })
    await assert.rejects(() => createFolderForIpc(path.join(root, 'missing', 'child')), { code: 'ENOENT' })
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})
