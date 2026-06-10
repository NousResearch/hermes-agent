'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')
const { pathToFileURL } = require('node:url')

const { readDirForIpc } = require('./fs-read-dir.cjs')

function mkTmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-fs-read-dir-'))
}

function fakeDirent(name, flags = {}) {
  return {
    name,
    isDirectory: () => Boolean(flags.directory),
    isFile: () => Boolean(flags.file),
    isSymbolicLink: () => Boolean(flags.symlink)
  }
}

test('readDirForIpc hides noisy directories and files from the project tree', async () => {
  const root = mkTmpDir()

  try {
    fs.mkdirSync(path.join(root, 'node_modules'))
    fs.mkdirSync(path.join(root, 'src'))
    fs.writeFileSync(path.join(root, 'target'), 'hidden file')
    fs.writeFileSync(path.join(root, 'README.md'), 'visible file')

    const result = await readDirForIpc(root)

    assert.equal(result.error, undefined)
    assert.deepEqual(
      result.entries.map(entry => entry.name),
      ['src', 'README.md']
    )
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('readDirForIpc filters a hidden basename whether it is a file or directory', async () => {
  const dirRoot = mkTmpDir()
  const fileRoot = mkTmpDir()

  try {
    fs.mkdirSync(path.join(dirRoot, 'node_modules'))
    fs.writeFileSync(path.join(dirRoot, 'visible.txt'), 'visible')
    fs.writeFileSync(path.join(fileRoot, 'node_modules'), 'hidden file')
    fs.writeFileSync(path.join(fileRoot, 'visible.txt'), 'visible')

    assert.deepEqual(
      (await readDirForIpc(dirRoot)).entries.map(entry => entry.name),
      ['visible.txt']
    )
    assert.deepEqual(
      (await readDirForIpc(fileRoot)).entries.map(entry => entry.name),
      ['visible.txt']
    )
  } finally {
    fs.rmSync(dirRoot, { recursive: true, force: true })
    fs.rmSync(fileRoot, { recursive: true, force: true })
  }
})

test('readDirForIpc returns directories before files and sorts by name within groups', async () => {
  const root = mkTmpDir()

  try {
    fs.writeFileSync(path.join(root, 'z.txt'), 'z')
    fs.mkdirSync(path.join(root, 'src'))
    fs.writeFileSync(path.join(root, 'a.txt'), 'a')
    fs.mkdirSync(path.join(root, 'lib'))

    const result = await readDirForIpc(root)

    assert.equal(result.error, undefined)
    assert.deepEqual(
      result.entries.map(entry => entry.name),
      ['lib', 'src', 'a.txt', 'z.txt']
    )
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('readDirForIpc accepts file URLs for directories', async () => {
  const root = mkTmpDir()

  try {
    fs.mkdirSync(path.join(root, 'src'))
    fs.writeFileSync(path.join(root, 'README.md'), 'visible file')

    const result = await readDirForIpc(pathToFileURL(root).toString())

    assert.equal(result.error, undefined)
    assert.deepEqual(
      result.entries.map(entry => entry.name),
      ['src', 'README.md']
    )
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('readDirForIpc returns invalid-path for blank or non-string input', async () => {
  let readdirCalls = 0
  const fsImpl = {
    promises: {
      readdir: async () => {
        readdirCalls += 1
        return []
      }
    }
  }

  assert.deepEqual(await readDirForIpc('', { fs: fsImpl }), { entries: [], error: 'invalid-path' })
  assert.deepEqual(await readDirForIpc('   ', { fs: fsImpl }), { entries: [], error: 'invalid-path' })
  assert.deepEqual(await readDirForIpc(null, { fs: fsImpl }), { entries: [], error: 'invalid-path' })
  assert.equal(readdirCalls, 0)
})

test('readDirForIpc rejects Windows device paths before readdir', async () => {
  let readdirCalls = 0
  const fsImpl = {
    promises: {
      readdir: async () => {
        readdirCalls += 1
        return []
      }
    }
  }

  assert.deepEqual(await readDirForIpc('\\\\?\\C:\\secret', { fs: fsImpl }), {
    entries: [],
    error: 'device-path'
  })
  assert.equal(readdirCalls, 0)
})

test('readDirForIpc returns filesystem error codes instead of throwing', async () => {
  const root = mkTmpDir()

  try {
    const result = await readDirForIpc(path.join(root, 'missing'))

    assert.deepEqual(result, { entries: [], error: 'ENOENT' })
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('readDirForIpc marks a symlink to a directory as a directory', async t => {
  const root = mkTmpDir()

  try {
    fs.mkdirSync(path.join(root, 'actual-dir'))

    try {
      fs.symlinkSync(path.join(root, 'actual-dir'), path.join(root, 'linked-dir'), 'dir')
    } catch (error) {
      if (error?.code === 'EPERM' || error?.code === 'EACCES') {
        t.skip(`symlink creation is not permitted on this platform (${error.code})`)

        return
      }

      throw error
    }

    const result = await readDirForIpc(root)
    const linked = result.entries.find(entry => entry.name === 'linked-dir')

    assert.equal(result.error, undefined)
    assert.equal(linked?.isDirectory, true)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('readDirForIpc marks a Windows junction to a directory as a directory', async t => {
  if (process.platform !== 'win32') {
    t.skip('junctions are a Windows-specific symlink type')

    return
  }

  const root = mkTmpDir()

  try {
    fs.mkdirSync(path.join(root, 'actual-dir'))

    try {
      fs.symlinkSync(path.join(root, 'actual-dir'), path.join(root, 'junction-dir'), 'junction')
    } catch (error) {
      if (error?.code === 'EPERM' || error?.code === 'EACCES') {
        t.skip(`junction creation is not permitted on this platform (${error.code})`)

        return
      }

      throw error
    }

    const result = await readDirForIpc(root)
    const junction = result.entries.find(entry => entry.name === 'junction-dir')

    assert.equal(result.error, undefined)
    assert.equal(junction?.isDirectory, true)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('readDirForIpc allows expanding symlink or junction directories outside the project root', async t => {
  const root = mkTmpDir()
  const outside = mkTmpDir()

  try {
    fs.writeFileSync(path.join(outside, 'outside.txt'), 'ok')

    const linkPath = path.join(root, 'outside-link')
    try {
      fs.symlinkSync(outside, linkPath, process.platform === 'win32' ? 'junction' : 'dir')
    } catch (error) {
      if (error?.code === 'EPERM' || error?.code === 'EACCES') {
        t.skip(`directory symlink creation is not permitted on this platform (${error.code})`)

        return
      }

      throw error
    }

    const result = await readDirForIpc(linkPath)

    assert.equal(result.error, undefined)
    assert.deepEqual(result.entries, [
      { name: 'outside.txt', path: path.join(linkPath, 'outside.txt'), isDirectory: false }
    ])
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
    fs.rmSync(outside, { recursive: true, force: true })
  }
})

test('readDirForIpc stats symbolic links and unknown entries without dropping the whole listing', async () => {
  const input = path.join('virtual-root')
  const resolved = path.resolve(input)
  const statCalls = []
  const fsImpl = {
    promises: {
      readdir: async () => [
        fakeDirent('unknown-entry'),
        fakeDirent('linked-dir', { symlink: true }),
        fakeDirent('broken-link', { symlink: true }),
        fakeDirent('plain.txt', { file: true })
      ],
      stat: async fullPath => {
        if (fullPath === resolved) {
          return { isDirectory: () => true }
        }

        statCalls.push(fullPath)
        if (fullPath.endsWith(`${path.sep}linked-dir`)) {
          return { isDirectory: () => true }
        }
        throw Object.assign(new Error('gone'), { code: 'ENOENT' })
      }
    }
  }

  const result = await readDirForIpc(input, { fs: fsImpl })

  assert.equal(result.error, undefined)
  assert.deepEqual(
    statCalls.sort(),
    [path.join(resolved, 'broken-link'), path.join(resolved, 'linked-dir'), path.join(resolved, 'unknown-entry')].sort()
  )
  assert.deepEqual(result.entries, [
    { name: 'linked-dir', path: path.join(resolved, 'linked-dir'), isDirectory: true },
    { name: 'broken-link', path: path.join(resolved, 'broken-link'), isDirectory: false },
    { name: 'plain.txt', path: path.join(resolved, 'plain.txt'), isDirectory: false },
    { name: 'unknown-entry', path: path.join(resolved, 'unknown-entry'), isDirectory: false }
  ])
})
