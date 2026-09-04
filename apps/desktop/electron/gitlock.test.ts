import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'

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

/**
 * Windows regression for #96694.
 *
 * gitProcessRunning() (private, driven here through clearStaleGitLocks with
 * the real probe enabled) spawns `tasklist` via execFile on win32. The caller
 * is the GUI-subsystem Electron main process, so the child MUST be spawned
 * with windowsHide: true — otherwise Windows allocates a visible console
 * window for it and the user sees a console flash on every update-check poll
 * with zero user activity. The existing behavioural tests above inject a fake
 * isGitRunning and therefore never exercise the real spawn; this block mocks
 * node:child_process and asserts on the actual options argument, mirroring
 * wsl-path-bridge-gate.test.ts.
 */
const execFileMock = vi.hoisted(() =>
  vi.fn(
    (
      _cmd: string,
      _args: readonly string[],
      _opts: Record<string, unknown>,
      cb: (err: unknown, stdout: string) => void
    ) => {
      // No git.exe in the output, so the probe resolves "not running" and the
      // stale-lock sweep proceeds end to end.
      cb(null, 'INFO: No tasks are running which match the specified criteria.')

      return undefined
    }
  )
)

vi.mock('node:child_process', () => ({ execFile: execFileMock }))

describe('gitProcessRunning windowsHide on Windows (#96694)', () => {
  const realPlatform = process.platform

  beforeEach(() => {
    Object.defineProperty(process, 'platform', { value: 'win32', configurable: true })
    execFileMock.mockClear()
  })

  afterEach(() => {
    Object.defineProperty(process, 'platform', { value: realPlatform, configurable: true })
  })

  test('the tasklist probe passes windowsHide: true so no console window flashes', async () => {
    const root = makeRepo()
    const lock = writeLock(root, 'shallow.lock', STALE_LOCK_MIN_AGE_MS + 60_000)

    const removed = await clearStaleGitLocks(root)

    // End-to-end: probe said "not running", so the stale lock was cleared.
    assert.deepEqual(removed, [lock])

    // The actual spawn options carried windowsHide.
    expect(execFileMock).toHaveBeenCalledTimes(1)
    const [cmd, cmdArgs, options] = execFileMock.mock.calls[0]
    assert.equal(options.windowsHide, true)
    // Sanity: it really was the tasklist probe, not some other child.
    assert.equal(cmd, 'tasklist')
    assert.deepEqual(cmdArgs, ['/FI', 'IMAGENAME eq git.exe', '/FO', 'CSV'])
  })

  test('the probe options are otherwise unchanged (timeout preserved)', async () => {
    const root = makeRepo()
    await clearStaleGitLocks(root)

    const [, , options] = execFileMock.mock.calls[0]
    assert.equal(options.timeout, 10_000)
  })
})
