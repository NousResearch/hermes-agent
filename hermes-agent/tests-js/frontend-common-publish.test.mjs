import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'

// Windows antivirus holds a freshly written tree (the HUD helper exe in
// native-deps, web_dist after a long build) and publication's renames fail
// EPERM until the scan ends (#126914). Fault only the renames a test names;
// every other rename reaches the real filesystem.
const faults = vi.hoisted(() => [])

vi.mock('node:fs', async importOriginal => {
  const actual = await importOriginal()
  return {
    ...actual,
    renameSync(from, to) {
      const fault = faults.find(entry => entry.match(from))
      if (fault) fault.calls++
      if (fault && fault.calls <= fault.failures) {
        throw Object.assign(new Error(`${fault.code}: rename '${from}' -> '${to}'`), { code: fault.code })
      }
      return actual.renameSync(from, to)
    },
  }
})

const { withProduct } = await import('../scripts/build/frontend-common.mjs')

let root

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'publish-held-'))
  vi.spyOn(Atomics, 'wait').mockReturnValue('timed-out') // the backoff is real; tests don't spend it
})

afterEach(() => {
  faults.length = 0
  vi.restoreAllMocks()
  rmSync(root, { recursive: true, force: true })
})

function fault(match, code, failures = Infinity) {
  const entry = { match: typeof match === 'string' ? from => from === match : match, code, failures, calls: 0 }
  faults.push(entry)
  return entry
}

function previousProduct() {
  const out = join(root, 'native-deps')
  mkdirSync(out)
  writeFileSync(join(out, '.hermes-product'), 'hermes-frontend-product-v1\n')
  writeFileSync(join(out, 'old.txt'), 'old')
  return out
}

test('a held staging tree publishes once the scanner lets go', async () => {
  const out = previousProduct()
  let staged
  await withProduct(out, product => {
    staged = fault(product, 'EPERM', 2)
    writeFileSync(join(product, 'hud-modifier-monitor.exe'), 'new')
  })
  expect(staged.calls).toBe(3)
  expect(readdirSync(out).sort()).toEqual(['.hermes-product', 'hud-modifier-monitor.exe'])
  expect(readdirSync(root)).toEqual(['native-deps'])
})

test('a code waiting cannot fix fails at once and keeps the previous product', async () => {
  const out = previousProduct()
  let staged
  await expect(withProduct(out, product => { staged = fault(product, 'EINVAL') })).rejects.toThrow(/EINVAL/)
  expect(staged.calls).toBe(1)
  expect(readFileSync(join(out, 'old.txt'), 'utf8')).toBe('old')
  expect(readdirSync(root)).toEqual(['native-deps'])
})

test('a hold that outlasts the budget rolls back to the previous product', async () => {
  const out = previousProduct()
  await expect(withProduct(out, product => { fault(product, 'EACCES') })).rejects.toThrow(/EACCES/)
  expect(readFileSync(join(out, 'old.txt'), 'utf8')).toBe('old')
  expect(readdirSync(root)).toEqual(['native-deps'])
})

test('a failed rollback survives scratch cleanup and names where the previous product is', async () => {
  const out = previousProduct()
  fault(from => from.includes('.native-deps-previous-'), 'EBUSY')
  const error = await withProduct(out, product => { fault(product, 'EPERM') }).catch(caught => caught)
  const backup = error.message.match(/previous product kept at (.+)$/)?.[1]
  expect(error.code).toBe('EPERM')
  expect(readFileSync(join(backup, 'old.txt'), 'utf8')).toBe('old')
})
