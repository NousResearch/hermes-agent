import fs from 'node:fs'
import path from 'node:path'

import { resolveDirectoryForIpc } from './hardening'

const FS_READDIR_STAT_CONCURRENCY = 16

// Always-hidden noise (covers non-git projects too; gitignore catches many of
// these, but the project tree should keep the same hygiene without one).
const FS_READDIR_HIDDEN = new Set([
  '.git',
  '.hg',
  '.svn',
  '.cache',
  '.next',
  '.turbo',
  '.venv',
  '__pycache__',
  'build',
  'dist',
  'node_modules',
  'target',
  'venv'
])

function direntIsDirectory(dirent) {
  return typeof dirent.isDirectory === 'function' && dirent.isDirectory()
}

function direntIsFile(dirent) {
  return typeof dirent.isFile === 'function' && dirent.isFile()
}

function direntIsSymbolicLink(dirent) {
  return typeof dirent.isSymbolicLink === 'function' && dirent.isSymbolicLink()
}

function shouldStatDirent(dirent) {
  if (direntIsDirectory(dirent)) {return false}

  return direntIsSymbolicLink(dirent) || !direntIsFile(dirent)
}

async function entryForDirent(dirent, resolved, fsImpl) {
  const fullPath = path.join(resolved, dirent.name)
  let isDirectory = direntIsDirectory(dirent)

  if (!isDirectory && shouldStatDirent(dirent)) {
    try {
      isDirectory = (await fsImpl.promises.stat(fullPath)).isDirectory()
    } catch {
      isDirectory = false
    }
  }

  return { name: dirent.name, path: fullPath, isDirectory }
}

async function mapWithStatConcurrency(items, mapper) {
  const results = new Array(items.length)
  let nextIndex = 0

  async function runWorker() {
    while (nextIndex < items.length) {
      const index = nextIndex
      nextIndex += 1
      results[index] = await mapper(items[index])
    }
  }

  const workerCount = Math.min(FS_READDIR_STAT_CONCURRENCY, items.length)
  const workers = Array.from({ length: workerCount } as any, () => runWorker())
  await Promise.all(workers)

  return results
}

async function readDirForIpc(dirPath, options: any = {}) {
  const fsImpl = options.fs || fs
  let resolved

  try {
    ;({ resolvedPath: resolved } = await resolveDirectoryForIpc(dirPath, {
      fs: fsImpl,
      purpose: 'Directory read'
    }))
  } catch (error) {
    return { entries: [], error: error?.code || 'read-error' }
  }

  try {
    const dirents = await fsImpl.promises.readdir(resolved, { withFileTypes: true })
    const visibleDirents = dirents.filter(dirent => !FS_READDIR_HIDDEN.has(dirent.name))
    const entries = await mapWithStatConcurrency(visibleDirents, dirent => entryForDirent(dirent, resolved, fsImpl))

    entries.sort((a, b) => Number(b.isDirectory) - Number(a.isDirectory) || a.name.localeCompare(b.name))

    return { entries }
  } catch (error) {
    return { entries: [], error: error?.code || 'read-error' }
  }
}

export { readDirForIpc }
