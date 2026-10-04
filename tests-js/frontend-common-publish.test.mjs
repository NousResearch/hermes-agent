import { existsSync, mkdtempSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, expect, test } from 'vitest'

import { PUBLISH_RENAME_ATTEMPTS, PUBLISH_RENAME_BACKOFF_MS, publishDirectory, withProduct } from '../scripts/build/frontend-common.mjs'

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

// Exercise real withProduct scratch cleanup and real filesystem renames; only
// inject faults at the three rename boundaries, never replace publication.
function lifecycle(out, faults = {}) {
  const calls = { backup: 0, publish: 0, restore: 0 }
  const sleeps = []
  const errors = {}
  let scratch
  let staged
  let backup
  return {
    calls, sleeps, errors,
    get scratch() { return scratch },
    get backup() { return backup },
    compile(product, work) {
      staged = product
      scratch = work
      writeFileSync(join(product, 'new.txt'), 'new')
    },
    deps: {
      sleep(ms) { sleeps.push(ms) },
      rename(from, to) {
        const seam = from === out ? 'backup' : from === staged ? 'publish' : 'restore'
        if (seam === 'backup') backup = to
        calls[seam] += 1
        const fault = faults[seam]
        if (fault && calls[seam] <= (fault.count ?? Infinity)) {
          fault.before?.(calls[seam])
          const error = new Error(`${fault.code}: ${seam} '${from}' -> '${to}'`)
          error.code = fault.code
          errors[seam] = error
          throw error
        }
        renameSync(from, to)
      },
    },
  }
}

function expectOld(out) {
  expect(readdirSync(out).sort()).toEqual(['.hermes-product', 'old.txt'])
  expect(readFileSync(join(out, 'old.txt'), 'utf8')).toBe('old')
  expect(readFileSync(join(out, '.hermes-product'), 'utf8')).toBe(PRODUCT_OWNER)
}

function expectSleeps(run, count) {
  expect(run.sleeps).toEqual(Array(count).fill(PUBLISH_RENAME_BACKOFF_MS))
}

for (const code of ['EPERM', 'EACCES', 'EBUSY']) {
  for (const seam of ['backup', 'publish', 'restore']) {
    test(`${code} transient ${seam} retries through withProduct`, async () => {
      const { root, out } = fixture(`${code}-${seam}`)
      const faults = { [seam]: { code, count: 2 } }
      if (seam === 'restore') faults.publish = { code: 'EINVAL' }
      const run = lifecycle(out, faults)
      if (seam === 'restore') {
        await expect(withProduct(out, run.compile, run.deps)).rejects.toThrow(/EINVAL: publish/)
        expectOld(out)
      } else {
        await withProduct(out, run.compile, run.deps)
        expect(readFileSync(join(out, 'new.txt'), 'utf8')).toBe('new')
        expect(existsSync(join(out, 'old.txt'))).toBe(false)
      }
      expect(run.calls[seam]).toBe(3)
      expectSleeps(run, 2)
      expect(existsSync(run.scratch)).toBe(false)
      expect(readdirSync(root).filter(name => name.includes('-previous-'))).toEqual([])
    })
  }

  test(`${code} exhausted live-product move leaves old output and cleans scratch`, async () => {
    const { root, out } = fixture(`${code}-backup-exhausted`)
    const run = lifecycle(out, { backup: { code } })
    let caught
    try { await withProduct(out, run.compile, run.deps) } catch (error) { caught = error }
    expect(caught).toBe(run.errors.backup)
    expect(run.calls).toEqual({ backup: PUBLISH_RENAME_ATTEMPTS, publish: 0, restore: 0 })
    expectSleeps(run, PUBLISH_RENAME_ATTEMPTS - 1)
    expectOld(out)
    expect(existsSync(run.scratch)).toBe(false)
    expect(readdirSync(root).filter(name => name.includes('-previous-'))).toEqual([])
  })

  test(`${code} exhausted publication retries transient rollback before cleanup`, async () => {
    const { out } = fixture(`${code}-publish-exhausted`)
    const run = lifecycle(out, { publish: { code }, restore: { code, count: 2 } })
    let caught
    try { await withProduct(out, run.compile, run.deps) } catch (error) { caught = error }
    expect(caught).toBe(run.errors.publish)
    expect(caught.rollbackError).toBeUndefined()
    expect(run.calls).toEqual({ backup: 1, publish: PUBLISH_RENAME_ATTEMPTS, restore: 3 })
    expectSleeps(run, PUBLISH_RENAME_ATTEMPTS - 1 + 2)
    expectOld(out)
    expect(existsSync(run.scratch)).toBe(false)
    expect(existsSync(run.backup)).toBe(false)
  })

  test(`${code} exhausted publication and rollback preserve diagnostics and old files after cleanup`, async () => {
    const { root, out } = fixture(`${code}-double-exhausted`)
    const run = lifecycle(out, {
      publish: { code, before(call) {
        if (call !== 1) return
        mkdirSync(out)
        writeFileSync(join(out, '.hermes-product'), PRODUCT_OWNER)
        writeFileSync(join(out, 'occupant.txt'), 'occupied')
      } },
      restore: { code },
    })
    let caught
    try { await withProduct(out, run.compile, run.deps) } catch (error) { caught = error }
    expect(caught).toBe(run.errors.publish)
    expect(caught.code).toBe(code)
    expect(caught.message).toContain(`${code}: publish`)
    expect(caught.message).toContain(`${code}: restore`)
    expect(caught.rollbackError).toBe(run.errors.restore)
    expect(caught.backupPath).toBe(run.backup)
    expect(caught.message).toContain(caught.backupPath)
    expect(run.calls).toEqual({ backup: 1, publish: PUBLISH_RENAME_ATTEMPTS, restore: PUBLISH_RENAME_ATTEMPTS })
    expectSleeps(run, 2 * (PUBLISH_RENAME_ATTEMPTS - 1))
    expect(existsSync(run.scratch)).toBe(false)
    expectOld(caught.backupPath)
    expect(readFileSync(join(out, 'occupant.txt'), 'utf8')).toBe('occupied')
    // Retained backups must also survive a subsequent successful build.
    await withProduct(out, product => writeFileSync(join(product, 'new.txt'), 'later'))
    expect(readFileSync(join(out, 'new.txt'), 'utf8')).toBe('later')
    expectOld(caught.backupPath)
    expect(readdirSync(root).filter(name => name.includes('-previous-'))).toHaveLength(1)
  })
}

for (const seam of ['backup', 'publish', 'restore']) {
  test(`non-transient ${seam} error does not retry or lose the old product`, async () => {
    const { out } = fixture(`nontransient-${seam}`)
    const faults = { [seam]: { code: 'EINVAL' } }
    if (seam === 'restore') faults.publish = { code: 'ENOENT' }
    const run = lifecycle(out, faults)
    let caught
    try { await withProduct(out, run.compile, run.deps) } catch (error) { caught = error }
    expect(run.calls[seam]).toBe(1)
    expectSleeps(run, 0)
    expect(existsSync(run.scratch)).toBe(false)
    expect(caught).toBe(run.errors[seam === 'restore' ? 'publish' : seam])
    if (seam === 'restore') {
      expect(caught.rollbackError).toBe(run.errors.restore)
      expectOld(caught.backupPath)
    } else expectOld(out)
  })
}

test('first publication uses real filesystem without creating a recovery backup', async () => {
  const { root } = fixture('first')
  const out = join(root, 'first-product')
  await withProduct(out, product => writeFileSync(join(product, 'new.txt'), 'first'))
  expect(readFileSync(join(out, 'new.txt'), 'utf8')).toBe('first')
  expect(readFileSync(join(out, '.hermes-product'), 'utf8')).toBe(PRODUCT_OWNER)
  expect(readdirSync(root).filter(name => name.startsWith('.first-product-'))).toEqual([])
})

test('compiler failure keeps old product and removes its real scratch tree', async () => {
  const { out } = fixture('compiler-failure')
  let scratch
  const failure = new Error('compiler failed')
  await expect(withProduct(out, (product, work) => {
    scratch = work
    writeFileSync(join(product, 'partial.txt'), 'partial')
    throw failure
  })).rejects.toBe(failure)
  expectOld(out)
  expect(existsSync(scratch)).toBe(false)
})
