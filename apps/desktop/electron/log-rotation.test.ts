import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  LOG_DISCARD_BYTES,
  LOG_MAX_BYTES,
  logBackupPath,
  planLogRotation,
  reclaimActiveLogIfOversized
} from './log-rotation'

// Regression for #100573 follow-up: the Chromium diagnostic log added for that
// issue is opened with APPEND_TO_OLD_LOG_FILE, so it grows across launches the
// same way desktop.log did before it was bounded (~326 GB, disk exhausted).
// The bound is one shared planner, so any log the shell keeps gets it.

test('a log under the cap is left alone, whatever its path', () => {
  assert.deepEqual(planLogRotation(LOG_MAX_BYTES - 1, '/logs/desktop-chromium.log'), [])
})

test('an oversized log cascades to backups instead of growing forever', () => {
  const base = '/logs/desktop-chromium.log'
  const ops = planLogRotation(LOG_MAX_BYTES, base)

  // The live file is moved aside, so the next launch starts from zero.
  assert.ok(ops.some(([op, src, dst]) => op === 'mv' && src === base && dst === logBackupPath(base, 1)))
  // The chain is bounded: the oldest backup is dropped, never accumulated.
  assert.deepEqual(ops[0], ['rm', logBackupPath(base, 3)])
  assert.ok(ops.every(([, src, dst]) => [src, dst].every(p => p === undefined || p.startsWith(base))))
})

test('a boot-loop log past the discard ceiling is reclaimed, not stranded in .1', () => {
  const base = '/logs/desktop-chromium.log'
  const ops = planLogRotation(LOG_DISCARD_BYTES + 1, base)

  // Renaming a multi-GB file keeps the disk full for a cycle a healthy app may
  // never reach, so every generation is deleted outright.
  assert.ok(ops.every(([op]) => op === 'rm'))
  assert.ok(ops.some(([, src]) => src === base))
})

test('a log a live process keeps appending to is reclaimed in place, not renamed', () => {
  const truncated: string[] = []
  const io = { size: () => LOG_MAX_BYTES, truncate: (f: string) => truncated.push(f) }

  // Chromium holds --log-file open in append mode for the life of the shell:
  // renaming it would leave the writer on the renamed inode and the cap would
  // silently stop applying, so the only reclamation is truncating in place.
  assert.equal(reclaimActiveLogIfOversized('/logs/desktop-chromium.log', io), true)
  assert.deepEqual(truncated, ['/logs/desktop-chromium.log'])
})

test('an under-cap or absent active log is left alone', () => {
  const touched: string[] = []
  const truncate = (f: string) => touched.push(f)

  assert.equal(reclaimActiveLogIfOversized('/logs/x.log', { size: () => LOG_MAX_BYTES - 1, truncate }), false)
  assert.equal(reclaimActiveLogIfOversized('/logs/x.log', { size: () => null, truncate }), false)
  assert.deepEqual(touched, [])
})

test('truncation really frees the file, and an append-mode writer restarts at 0', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-log-bound-'))
  const file = path.join(dir, 'desktop-chromium.log')

  try {
    // Windows denies ftruncate on an append-only handle. Create the sparse
    // oversized log before Chromium's stand-in opens it for appending.
    const seed = fs.openSync(file, 'w+')

    try {
      fs.ftruncateSync(seed, LOG_MAX_BYTES + 1) // Grow without writing GBs.
    } finally {
      fs.closeSync(seed)
    }

    // Stand in for Chromium: an O_APPEND handle held across the reclaim.
    const handle = fs.openSync(file, 'a')

    try {
      assert.equal(
        reclaimActiveLogIfOversized(file, {
          size: f => fs.statSync(f).size,
          truncate: f => fs.truncateSync(f, 0)
        }),
        true
      )

      fs.writeSync(handle, 'FATAL:after\n')
      assert.equal(fs.readFileSync(file, 'utf8'), 'FATAL:after\n')
    } finally {
      fs.closeSync(handle)
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})
