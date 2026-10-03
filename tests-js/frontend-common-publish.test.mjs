import { existsSync, mkdtempSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, expect, test } from 'vitest'

import { PUBLISH_RENAME_ATTEMPTS, PUBLISH_RENAME_BACKOFF_MS, publishDirectory } from '../scripts/build/frontend-common.mjs'

const roots = []

afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })

// Fixes #126914: the staged->out rename fails transiently on Windows while an
// antivirus scanner walks the fresh staging tree (Sophos EPERM -4048) or a
// just-drained process's handles close. publishDirectory must absorb short
// bursts and stay bounded on persistent locks.
const PRODUCT_OWNER = 'hermes-frontend-product-v1\n'

function fixture(tag) {
  const root = mkdtempSync(join(tmpdir(), 'publish-retry-' + tag + '-'))
  roots.push(root)
  const out = join(root, 'web_dist')
  mkdirSync(out)
  writeFileSync(join(out, '.hermes-product'), PRODUCT_OWNER)
  writeFileSync(join(out, 'old.txt'), 'old')
  const staged = join(root, '.web_dist-build-X')
  mkdirSync(staged)
  writeFileSync(join(staged, 'new.txt'), 'new')
  return { root, out, staged }
}

// Faults ONLY the staged->out rename; the out->backup and backup->out restore
// renames (different names) pass through to the real fs untouched.
function flakyRename(stagedPath, faultCount, faultForever) {
  const real = renameSync
  let calls = 0
  return (from, to) => {
    if (from === stagedPath) {
      calls += 1
      if (faultForever || calls <= faultCount) {
        const error = new Error(`EPERM: operation not permitted, rename '${from}' -> '${to}'`)
        error.code = 'EPERM'
        throw error
      }
    }
    return real(from, to)
  }
}
const realSleep = ms => { Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms) }
const tick = () => realSleep(50)

test('a persistent EPERM keeps the documented budget and rethrows', () => {
  const { out, staged } = fixture('persistent')
  const started = Date.now()
  expect(() => publishDirectory(staged, out, { rename: flakyRename(staged, 0, true), sleep: realSleep }))
    .toThrowError(/EPERM/)
  // 5 real backoffs of PUBLISH_RENAME_BACKOFF_MS; clock-bounded below, not a snapshot.
  expect(Date.now() - started).toBeGreaterThanOrEqual(4 * PUBLISH_RENAME_BACKOFF_MS)
  expect(PUBLISH_RENAME_ATTEMPTS).toBeGreaterThan(1)
  // publishDirectory restored the previous product before rethrowing.
  expect(readdirSync(out).sort()).toEqual(['.hermes-product', 'old.txt'])
})

test('non-transient errors rethrow immediately without burning the retry budget', () => {
  const { out, staged } = fixture('enoent')
  let calls = 0
  const rename = () => {
    calls += 1
    const error = new Error('ENOENT: no such file or directory')
    error.code = 'ENOENT'
    throw error
  }
  expect(() => publishDirectory(staged, out, { rename, sleep: tick })).toThrowError(/ENOENT/)
  expect(calls).toBe(1)
  expect(readdirSync(out).sort()).toEqual(['.hermes-product', 'old.txt'])
})

test('a transient burst shorter than the budget publishes the new product', () => {
  const { out, staged } = fixture('transient')
  const rename = flakyRename(staged, 2, false)
  publishDirectory(staged, out, { rename, sleep: tick })
  expect(readdirSync(out).sort()).toEqual(['.hermes-product', 'new.txt'])
  expect(readFileSync(join(out, '.hermes-product'), 'utf8')).toBe(PRODUCT_OWNER)
  expect(existsSync(staged)).toBe(false)
  expect(existsSync(`${staged}.previous`)).toBe(false)
})

test('a burst longer than the budget fails closed and restores the previous product', () => {
  const { out, staged } = fixture('overrun')
  expect(() => publishDirectory(staged, out, { rename: flakyRename(staged, 0, true), sleep: tick }))
    .toThrowError(/EPERM/)
  expect(readdirSync(out).sort()).toEqual(['.hermes-product', 'old.txt'])
  // The staged dir survives: publishDirectory leaves failure evidence in place.
  expect(existsSync(staged)).toBe(true)
})
