import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'

// A rebuild copies the previous generation's chunks for tabs that still load them; a copy can
// fail for reasons that are no fault of the entry (a full disk). Fault only the copies a test
// names, by the complete native path the build copies from; every other copy reaches the real
// filesystem.
const faults = vi.hoisted(() => [])

vi.mock('node:fs', async importOriginal => {
  const actual = await importOriginal()
  return {
    ...actual,
    copyFileSync(from, to, mode) {
      const fault = faults.find(entry => from === entry.from)
      if (fault) {
        if (fault.written !== undefined) actual.writeFileSync(to, fault.written)
        throw Object.assign(new Error(`${fault.code}: copyfile '${from}' -> '${to}'`), { code: fault.code })
      }
      return actual.copyFileSync(from, to, mode)
    },
  }
})

const { keepPreviousGeneration } = await import('../scripts/build/webapp.mjs')
const { buildInputs, recordProduct } = await import('../scripts/build/freshness.mjs')

let root
let warn

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'webapp-keep-'))
  warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
})

afterEach(() => {
  faults.length = 0
  vi.restoreAllMocks()
  rmSync(root, { recursive: true, force: true })
})

function put(dir, name, content) {
  mkdirSync(dirname(join(dir, name)), { recursive: true })
  writeFileSync(join(dir, name), content)
}

function generations(previousFiles, nextFiles, manifest) {
  const [previous, next] = [join(root, 'previous'), join(root, 'next')]
  for (const [name, content] of Object.entries(previousFiles)) put(previous, name, content)
  for (const [name, content] of Object.entries(nextFiles)) put(next, name, content)
  put(previous, 'renderer-manifest.json', JSON.stringify(manifest))
  recordProduct({ source: root, product: 'webapp', out: previous, inputs: buildInputs(root, 'webapp') })
  return { previous, next }
}

test('a chunk that cannot be copied is reported with its path, and the rest are still kept', () => {
  const { previous, next } = generations(
    { 'index.html': 'old', 'assets/full.js': 'old full', 'assets/ok.js': 'old ok' },
    { 'index.html': 'new' },
    { a: { file: 'assets/full.js', assets: ['assets/ok.js'] } },
  )
  faults.push({ from: join(previous, 'assets/full.js'), code: 'ENOSPC', written: 'partial' })
  expect(keepPreviousGeneration(previous, next)).toEqual(['assets/ok.js'])
  expect(warn).toHaveBeenCalledTimes(1)
  expect(warn.mock.calls[0][0]).toContain(join(previous, 'assets/full.js'))
  expect(warn.mock.calls[0][0]).toContain('ENOSPC')
  // The failed copy's partial file is not published as the chunk.
  expect(existsSync(join(next, 'assets/full.js'))).toBe(false)
  expect(readFileSync(join(next, 'assets/ok.js'), 'utf8')).toBe('old ok')
})

test('a missing previous build, rejected entries and a new file standing in the way stay quiet', () => {
  expect(keepPreviousGeneration(join(root, 'never-built'), join(root, 'next'))).toEqual([])
  const { previous, next } = generations(
    { 'index.html': 'old', 'assets/raced.js': 'old raced', 'x/kept.js': 'old' },
    { 'index.html': 'new', 'x': 'new file' },
    { a: { file: 'assets/raced.js', css: ['../escape.css', '/abs.css'], assets: ['assets/gone.js', 'bad\0name.js', 'x/kept.js', 'index.html'] } },
  )
  // The new build wrote this name between the existence check and the copy.
  faults.push({ from: join(previous, 'assets/raced.js'), code: 'EEXIST', written: 'new raced' })
  expect(keepPreviousGeneration(previous, next)).toEqual([])
  expect(warn).not.toHaveBeenCalled()
  expect(readFileSync(join(next, 'assets/raced.js'), 'utf8')).toBe('new raced')
  expect(readFileSync(join(next, 'x'), 'utf8')).toBe('new file')
})
