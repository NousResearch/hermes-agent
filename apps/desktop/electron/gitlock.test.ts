import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { clearStaleGitLocks, LOCK_NAMES, STALE_LOCK_MIN_AGE_MS } from './gitlock'

function makeRepo(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'gitlock-test-'))
  fs.mkdirSync(path.join(root, '.git'))

  return root
}

function writeLock(root: string, name: string, ageMs: number): string {
  const p = path.join(root, '.git', name)
  fs.writeFileSync(p, '')
  const t = new Date(Date.now() - ageMs)
  fs.utimesSync(p, t, t)

  return p
}

const noGit = async () => false
const gitRunning = async () => true

test('stale shallow.lock older than min age is removed', async () => {
  const root = makeRepo()
  const lock = writeLock(root, 'shallow.lock', STALE_LOCK_MIN_AGE_MS + 60_000)
  const removed = await clearStaleGitLocks(root, { isGitRunning: noGit })
  assert.deepEqual(removed, [lock])
  assert.equal(fs.existsSync(lock), false)
})

test('fresh lock is presumed live and never removed', async () => {
  const root = makeRepo()
  const lock = writeLock(root, 'shallow.lock', 1_000)
  const removed = await clearStaleGitLocks(root, { isGitRunning: noGit })
  assert.deepEqual(removed, [])
  assert.equal(fs.existsSync(lock), true)
})

test('running git process protects even ancient locks', async () => {
  const root = makeRepo()
  const lock = writeLock(root, 'shallow.lock', STALE_LOCK_MIN_AGE_MS * 10)
  const removed = await clearStaleGitLocks(root, { isGitRunning: gitRunning })
  assert.deepEqual(removed, [])
  assert.equal(fs.existsSync(lock), true)
})

test('all known lock names are cleared when stale', async () => {
  const root = makeRepo()
  const locks = LOCK_NAMES.map(name => writeLock(root, name, STALE_LOCK_MIN_AGE_MS + 60_000))
  const removed = await clearStaleGitLocks(root, { isGitRunning: noGit })
  assert.deepEqual(removed.sort(), locks.sort())
})

test('unknown lock-like files are left alone', async () => {
  const root = makeRepo()
  const stray = writeLock(root, 'config.lock', STALE_LOCK_MIN_AGE_MS * 10)
  await clearStaleGitLocks(root, { isGitRunning: noGit })
  assert.equal(fs.existsSync(stray), true)
})

test('missing .git dir is a silent no-op', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'gitlock-nogit-'))
  const removed = await clearStaleGitLocks(root, { isGitRunning: noGit })
  assert.deepEqual(removed, [])
})
