import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { assertExistingPathForOpen } from './hardening'

// #122027: a deleted/renamed file handed to the OS comes back as
// "No application found to open URL" on the open path (LaunchServices
// kLSApplicationNotFoundErr doubles as the missing-file answer on macOS), and
// the current reveal-in-folder route silently no-ops on a missing path — the
// click does nothing at all. The guard must say "missing" before the OS is
// asked, on all platforms — fs.statSync's ENOENT/ENOTDIR are uniform across
// macOS/Windows/Linux.

test('a missing path throws a missing-file error instead of reaching the OS', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'open-missing-'))
  const ghost = path.join(dir, 'report.html')

  assert.throws(
    () => assertExistingPathForOpen(ghost, 'Open preview in browser'),
    (error: Error & { code?: string }) => {
      assert.equal(error.code, 'missing-file')
      assert.match(error.message, /does not exist/)
      assert.match(error.message, /another machine/)

      return true
    }
  )
})

test('a path under a missing directory (ENOTDIR) also reads as missing', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'open-missing-'))
  const underFile = path.join(path.join(dir, 'plain.txt'), 'nested.txt')

  fs.writeFileSync(path.join(dir, 'plain.txt'), 'not a directory')

  assert.throws(
    () => assertExistingPathForOpen(underFile),
    (error: Error & { code?: string }) => error.code === 'missing-file'
  )
})

test('an existing file passes the guard', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'open-present-'))
  const real = path.join(dir, 'report.html')

  fs.writeFileSync(real, '<html></html>')

  assert.doesNotThrow(() => assertExistingPathForOpen(real, 'Open preview in browser'))
})

test('a non-ENOENT stat failure is not converted into a fabricated miss', () => {
  // The stat itself is the seam under test: an EACCES-style failure on an
  // EXISTING file must propagate untouched so the OS still gets a chance
  // (root-owned trees are openable by their owner's other tooling, and the
  // OS error there is honest — it is not "no application").
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'open-eacces-'))
  const real = path.join(dir, 'locked.txt')

  fs.writeFileSync(real, 'x')

  const eacces = Object.assign(new Error('EACCES: permission denied'), { code: 'EACCES' })
  const original = fs.statSync
  Object.defineProperty(fs, 'statSync', {
    configurable: true,
    value: (p: fs.PathLike) => {
      if (String(p) === real) {
        throw eacces
      }

      return original(p)
    }
  })

  try {
    assert.throws(
      () => assertExistingPathForOpen(real),
      (error: Error & { code?: string }) => error.code === 'EACCES'
    )
  } finally {
    Object.defineProperty(fs, 'statSync', { configurable: true, value: original })
  }
})

test('the purpose string reaches the user-facing message', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'open-purpose-'))

  assert.throws(
    () => assertExistingPathForOpen(path.join(dir, 'gone.pdf'), 'Open external file'),
    /Open external file failed/
  )
})
