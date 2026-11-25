import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, expect, test, vi } from 'vitest'
import { observeSourceUpdate } from '../tests/install/e2e-assets/source-update-observer.mjs'

// Lets a test play the relaunched app: it unlinks the result file after the
// observer listed it and before the observer reads it.
const hooks = vi.hoisted(() => ({ beforeRead: null }))
vi.mock('node:fs', async importOriginal => {
  const actual = await importOriginal()
  return { ...actual, readFileSync: (...args) => { hooks.beforeRead?.(args[0]); return actual.readFileSync(...args) } }
})

const roots = []
afterEach(() => { hooks.beforeRead = null; roots.splice(0).forEach(root => rmSync(root, { recursive: true, force: true })) })
function fixture() {
  const home = mkdtempSync(join(tmpdir(), 'source-update-observer-'))
  roots.push(home)
  const receipts = join(home, 'logs/update_receipts')
  mkdirSync(receipts, { recursive: true })
  const resultPath = join(home, '.hermes-update-result.json')
  const write = (name, data) => writeFileSync(join(receipts, name), JSON.stringify(data))
  return { home, resultPath, write }
}

test('checkout movement and old receipts are not update completion', () => {
  const f = fixture()
  f.write('update_old.json', { outcome: 'success', finished_at: 'old', post_update: { sha: 'target' } })
  const observe = observeSourceUpdate({ ...f, expectSha: 'target' })
  expect(observe('target')).toBe(false)
  f.write('pm_sync.json', { outcome: 'success', finished_at: 'now', post_update: { sha: 'target' } })
  expect(observe('target')).toBe(false)
  f.write('update_new.json', { outcome: 'success', finished_at: 'now', post_update: { sha: 'target' } })
  expect(observe('old')).toBe(false)
  expect(observe('target')).toBe(true)
})

test('failed handoff fails even at target SHA; success waits for marker removal', () => {
  const f = fixture()
  const observe = observeSourceUpdate({ ...f, expectSha: 'target' })
  writeFileSync(f.resultPath, JSON.stringify({ ok: false, exit_code: 1 }))
  expect(() => observe('target')).toThrow(/failed/)
  writeFileSync(f.resultPath, JSON.stringify({ ok: true, exit_code: 0 }))
  const marker = join(f.home, '.hermes-update-in-progress')
  writeFileSync(marker, 'updater still finishing')
  expect(observe('target')).toBe(false)
  rmSync(marker)
  expect(observe('target')).toBe(true)
})

test('a result file consumed by the relaunched app mid-poll does not abort the update', () => {
  const f = fixture()
  const observe = observeSourceUpdate({ ...f, expectSha: 'target' })
  f.write('update_new.json', { outcome: 'success', finished_at: 'now', post_update: { sha: 'target' } })
  writeFileSync(f.resultPath, JSON.stringify({ ok: true, exit_code: 0 }))
  hooks.beforeRead = file => { if (file === f.resultPath) rmSync(file) }
  expect(observe('target')).toBe(true)
})
