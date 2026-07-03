import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { scanGitRepos } from './git-repo-scan'

function mkTmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-git-repo-scan-'))
}

function mkRepo(root: string, ...segments: string[]) {
  const repo = path.join(root, ...segments)
  fs.mkdirSync(path.join(repo, '.git'), { recursive: true })

  return repo
}

function foundRoots(results: { root: string }[]) {
  return results.map(entry => entry.root).sort()
}

test('finds a normal repo but skips root-level macOS media folders', async () => {
  const root = mkTmpDir()

  try {
    const dev = mkRepo(root, 'dev', 'proj')
    mkRepo(root, 'Pictures', 'wallpapers')
    mkRepo(root, 'Music', 'samples')
    mkRepo(root, 'Movies', 'clips')
    mkRepo(root, 'Public', 'shared')

    assert.deepEqual(foundRoots(await scanGitRepos([root])), [dev])
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('still scans a media-named directory below the search root', async () => {
  const root = mkTmpDir()

  try {
    const nested = mkRepo(root, 'dev', 'Music', 'app')

    assert.deepEqual(foundRoots(await scanGitRepos([root])), [nested])
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('skips Apple media-library packages at any depth', async () => {
  const root = mkTmpDir()

  try {
    const keeper = mkRepo(root, 'code', 'site')
    mkRepo(root, 'code', 'Photos Library.photoslibrary', 'inner')
    mkRepo(root, 'backups', 'Music Library.MUSICLIBRARY', 'inner')
    mkRepo(root, 'backups', 'TV Library.tvlibrary', 'inner')
    mkRepo(root, 'backups', 'Old.aplibrary', 'inner')

    assert.deepEqual(foundRoots(await scanGitRepos([root])), [keeper])
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('walks an explicitly passed media root', async () => {
  const root = mkTmpDir()

  try {
    const musicRoot = path.join(root, 'Music')
    const repo = mkRepo(musicRoot, 'samples')

    assert.deepEqual(foundRoots(await scanGitRepos([musicRoot])), [repo])
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})
