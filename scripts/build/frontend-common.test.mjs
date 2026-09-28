import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, statSync, existsSync, renameSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { publishDirectory, renameSyncRetry } from './frontend-common.mjs'

function makeTree(tag) {
  const root = mkdtempSync(join(tmpdir(), `publish-retry-${tag}-`))
  const staged = join(root, 'product')
  mkdirSync(staged)
  writeFileSync(join(staged, 'a.txt'), 'x'.repeat(16))
  const out = join(root, 'dist')
  return { root, staged, out }
}

function eperm() {
  const err = new Error('EPERM: operation not permitted, rename')
  err.code = 'EPERM'
  return err
}

test('renameSyncRetry succeeds through transient EPERM failures', () => {
  const { root, staged, out } = makeTree('transient')
  try {
    let calls = 0
    const rename = (from, to) => {
      calls += 1
      if (calls <= 2) throw eperm()
      renameSync(from, to)
    }
    publishDirectory(staged, out, { source: 'src/index.ts', rename })
    assert.ok(statSync(join(out, 'a.txt')).isFile(), 'published despite 2 transient failures')
    assert.ok(statSync(join(out, '.hermes-product')).isFile(), 'product marker written')
    assert.equal(calls, 3, 'two failures then one success')
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('renameSyncRetry gives up after the budget and throws the last EPERM', () => {
  const { root, staged, out } = makeTree('budget')
  try {
    const alwaysFail = () => { throw eperm() }
    assert.throws(
      () => renameSyncRetry(staged, out, { attempts: 3, delayMs: 1, rename: alwaysFail }),
      err => err.code === 'EPERM',
    )
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('non-retryable errors propagate immediately without retry', () => {
  const { root, staged, out } = makeTree('nonretry')
  try {
    let calls = 0
    const enoent = (from, to) => {
      calls += 1
      const err = new Error('ENOENT: no such file or directory')
      err.code = 'ENOENT'
      throw err
    }
    assert.throws(
      () => renameSyncRetry(staged, out, { attempts: 6, delayMs: 1, rename: enoent }),
      err => err.code === 'ENOENT',
    )
    assert.equal(calls, 1, 'no retries on non-retryable codes')
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('publishDirectory replaces a previous product and clears the backup', () => {
  const { root, staged, out } = makeTree('swap')
  try {
    publishDirectory(staged, out, { source: 'src/index.ts' })
    assert.ok(existsSync(join(out, 'a.txt')), 'first publish landed')
    const staged2 = join(root, 'product2')
    mkdirSync(staged2)
    writeFileSync(join(staged2, 'b.txt'), 'y'.repeat(16))
    publishDirectory(staged2, out, { source: 'src/index.ts' })
    assert.ok(existsSync(join(out, 'b.txt')), 'new product in place')
    assert.ok(!existsSync(join(out, 'a.txt')), 'old product replaced')
    assert.ok(!existsSync(join(root, 'product2.previous')), 'backup cleared')
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('a failing replace restores the previous product from the backup', () => {
  const { root, staged, out } = makeTree('restore')
  try {
    publishDirectory(staged, out, { source: 'src/index.ts' })
    assert.ok(existsSync(join(out, 'a.txt')), 'first publish landed')
    const staged2 = join(root, 'product2')
    mkdirSync(staged2)
    writeFileSync(join(staged2, 'b.txt'), 'y'.repeat(16))
    const failStagedToOut = (from, to) => from === staged2 && to === out
    const rename = (from, to) => {
      if (failStagedToOut(from, to)) throw eperm() // the only rename that must fail
      renameSync(from, to)                          // out -> backup, and the restore, succeed
    }
    assert.throws(
      () => publishDirectory(staged2, out, { source: 'src/index.ts', rename }),
      err => err.code === 'EPERM',
    )
    assert.ok(existsSync(join(out, 'a.txt')), 'previous product restored to out')
    assert.ok(!existsSync(join(out, 'b.txt')), 'failed product did not land')
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
