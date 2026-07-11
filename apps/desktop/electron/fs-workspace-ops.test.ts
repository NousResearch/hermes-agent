import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

import {
  createDirectoryForIpc,
  createFileForIpc,
  deletePathForIpc,
  movePathForIpc,
  renamePathForIpc
} from './fs-workspace-ops'

test('createDirectoryForIpc creates one folder and rejects collisions', async () => {
  const root = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-fs-'))

  try {
    assert.deepEqual(await createDirectoryForIpc(root, 'new'), { path: path.join(root, 'new') })
    await assert.rejects(() => createDirectoryForIpc(root, 'new'), /already exists/i)
    await assert.rejects(() => createDirectoryForIpc(root, '../escape'), /invalid/i)
  } finally {
    await fs.promises.rm(root, { force: true, recursive: true })
  }
})

test('createFileForIpc creates exclusively and rejects traversal names', async () => {
  const root = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-fs-'))

  try {
    assert.deepEqual(await createFileForIpc(root, 'notes.md'), { path: path.join(root, 'notes.md') })
    await assert.rejects(() => createFileForIpc(root, 'notes.md'), /already exists/i)
    await assert.rejects(() => createFileForIpc(root, '../escape'), /invalid/i)
  } finally {
    await fs.promises.rm(root, { force: true, recursive: true })
  }
})

test('renamePathForIpc hardens names and rejects collisions', async () => {
  const root = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-fs-'))

  try {
    const source = path.join(root, 'a.txt')
    await fs.promises.writeFile(source, 'a')
    await fs.promises.writeFile(path.join(root, 'exists.txt'), 'b')
    assert.deepEqual(await renamePathForIpc(source, 'b.txt'), { path: path.join(root, 'b.txt') })
    await assert.rejects(() => renamePathForIpc(path.join(root, 'b.txt'), 'exists.txt'), /already exists/i)
    await assert.rejects(() => renamePathForIpc(path.join(root, 'b.txt'), '..'), /invalid/i)
  } finally {
    await fs.promises.rm(root, { force: true, recursive: true })
  }
})

test('movePathForIpc rejects roots, descendants, and collisions', async () => {
  const root = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-fs-'))

  try {
    const source = path.join(root, 'folder')
    const destination = path.join(root, 'target')
    await fs.promises.mkdir(path.join(source, 'child'), { recursive: true })
    await fs.promises.mkdir(destination)
    await assert.rejects(() => movePathForIpc(source, path.join(source, 'child'), root), /itself|descendant/i)
    assert.deepEqual(await movePathForIpc(source, destination, root), { path: path.join(destination, 'folder') })
    await assert.rejects(() => movePathForIpc(root, destination, root), /browser root/i)
  } finally {
    await fs.promises.rm(root, { force: true, recursive: true })
  }
})

test('deletePathForIpc rejects browser roots and symlinks', async () => {
  const root = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-fs-'))

  try {
    const file = path.join(root, 'notes.md')
    const link = path.join(root, 'notes-link')
    await fs.promises.writeFile(file, 'notes')
    await fs.promises.symlink(file, link)
    assert.equal(await deletePathForIpc(file, root), file)
    await assert.rejects(() => deletePathForIpc(root, root), /browser root/i)
    await assert.rejects(() => deletePathForIpc(link, root), /symlink/i)
  } finally {
    await fs.promises.rm(root, { force: true, recursive: true })
  }
})
