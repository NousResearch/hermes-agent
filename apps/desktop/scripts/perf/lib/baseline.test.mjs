import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, test } from 'vitest'

import {
  baselinePlatforms,
  compareScenario,
  loadBaseline,
  platformKey,
  updateBaseline
} from './baseline.mjs'

let dir

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'hermes-baseline-test-'))
})

afterEach(() => {
  rmSync(dir, { recursive: true, force: true })
})

const pathIn = name => join(dir, name)

function writeDoc(name, doc) {
  const path = pathIn(name)

  writeFileSync(path, JSON.stringify(doc, null, 2))

  return path
}

const legacyDoc = () => ({
  _meta: { note: 'Median of 5 runs, darwin-arm64', platform: 'darwin-arm64', node: 'v24.11.0' },
  scenarios: {
    stream: { tolerance: { tolFrac: 0.6, tolAbs: 5 }, metrics: { frame_p95_ms: 22 } }
  }
})

function readBack(path) {
  return JSON.parse(readFileSync(path, 'utf8'))
}

test('platformKey composes platform and arch', () => {
  assert.equal(platformKey('win32', 'x64'), 'win32-x64')
  assert.equal(platformKey('darwin', 'arm64'), 'darwin-arm64')
})

test('baselinePlatforms reads the per-platform shape as-is', () => {
  const platforms = baselinePlatforms({ platforms: { 'win32-x64': { scenarios: { a: {} } } } })

  assert.deepEqual(Object.keys(platforms), ['win32-x64'])
})

test('baselinePlatforms migrates the legacy single-platform shape to its own bucket', () => {
  const platforms = baselinePlatforms(legacyDoc())

  // The committed numbers were captured on darwin-arm64, so they belong to that
  // bucket — not to whichever machine happens to read the file next.
  assert.deepEqual(Object.keys(platforms), ['darwin-arm64'])
  assert.equal(platforms['darwin-arm64'].scenarios.stream.metrics.frame_p95_ms, 22)
})

test('baselinePlatforms returns nothing for an empty or shapeless document', () => {
  assert.deepEqual(baselinePlatforms({}), {})
  assert.deepEqual(baselinePlatforms({ _meta: { platform: 'darwin-arm64' }, scenarios: {} }), {})
  assert.deepEqual(baselinePlatforms({ _meta: { platform: 'darwin-arm64' } }), {})
})

test('loadBaseline is gated for the platform it holds', () => {
  const path = writeDoc('b.json', { _meta: {}, platforms: { 'win32-x64': { scenarios: { stream: { metrics: {} } } } } })
  const loaded = loadBaseline(path, 'win32-x64')

  assert.equal(loaded.gated, true)
  assert.deepEqual(Object.keys(loaded.scenarios), ['stream'])
})

test('loadBaseline is NOT gated for a platform with no bucket of its own', () => {
  const path = writeDoc('b.json', { _meta: {}, platforms: { 'darwin-arm64': { scenarios: { stream: { metrics: {} } } } } })

  // The bug this guards: a Windows/Linux run used to compare against the Mac's
  // numbers. It must instead report its metrics and gate against nothing.
  const loaded = loadBaseline(path, 'win32-x64')

  assert.equal(loaded.gated, false)
  assert.deepEqual(loaded.scenarios, {})
  assert.deepEqual(loaded.availablePlatforms, ['darwin-arm64'])
})

test('loadBaseline survives a missing or corrupt file', () => {
  for (const path of [pathIn('nope.json'), writeDoc('bad.json', null)]) {
    const loaded = loadBaseline(path, 'win32-x64')

    assert.equal(loaded.gated, false)
    assert.deepEqual(loaded.scenarios, {})
  }
})

test('an ungated platform regresses on nothing', () => {
  const path = writeDoc('b.json', { _meta: {}, platforms: {} })
  const loaded = loadBaseline(path, 'win32-x64')
  const { rows, regressed } = compareScenario('stream', { frame_p95_ms: 900, longtasks_n: 12 }, loaded)

  assert.equal(regressed, false)
  assert.deepEqual(rows.map(r => r.status), ['new', 'new'])
})

test('a gated platform still gates', () => {
  const path = writeDoc('b.json', {
    _meta: {},
    platforms: { 'win32-x64': { scenarios: { stream: { tolerance: { tolFrac: 0.6, tolAbs: 5 }, metrics: { frame_p95_ms: 22 } } } } }
  })
  const loaded = loadBaseline(path, 'win32-x64')

  assert.equal(compareScenario('stream', { frame_p95_ms: 24 }, loaded).regressed, false)
  assert.equal(compareScenario('stream', { frame_p95_ms: 900 }, loaded).regressed, true)
})

test('updateBaseline writes only its own bucket and leaves other platforms intact', () => {
  const path = writeDoc('b.json', legacyDoc())
  const written = updateBaseline(path, [{ name: 'keystroke', metrics: { keystroke_p50_ms: 4, ignored: 'nope' } }], 'win32-x64')
  const doc = readBack(path)

  assert.equal(written.platform, 'win32-x64')
  assert.deepEqual(written.scenarios, ['keystroke'])

  // The Mac's bucket is untouched — that is the whole point.
  assert.equal(doc.platforms['darwin-arm64'].scenarios.stream.metrics.frame_p95_ms, 22)
  assert.equal(doc.platforms['win32-x64'].scenarios.keystroke.metrics.keystroke_p50_ms, 4)
  assert.deepEqual(doc._meta.platforms, ['darwin-arm64', 'win32-x64'])
})

test('updateBaseline drops non-numeric metrics and the legacy platform label', () => {
  const path = writeDoc('b.json', legacyDoc())

  updateBaseline(path, [{ name: 'stream', metrics: { frame_p95_ms: 30, note: 'text', flag: true } }], 'win32-x64')
  const doc = readBack(path)

  assert.deepEqual(doc.platforms['win32-x64'].scenarios.stream.metrics, { frame_p95_ms: 30 })
  assert.equal('platform' in doc._meta, false)
  assert.equal(doc._meta.note, 'Median of 5 runs, darwin-arm64')
})

test('updateBaseline merges within a bucket instead of dropping sibling scenarios', () => {
  const path = writeDoc('b.json', {
    _meta: {},
    platforms: {
      'win32-x64': {
        scenarios: {
          stream: { tolerance: { tolFrac: 0.9 }, metrics: { frame_p95_ms: 22 } },
          keystroke: { metrics: { keystroke_p50_ms: 3 } }
        }
      }
    }
  })

  updateBaseline(path, [{ name: 'stream', metrics: { frame_p95_ms: 25 } }], 'win32-x64')
  const doc = readBack(path)

  assert.equal(doc.platforms['win32-x64'].scenarios.stream.metrics.frame_p95_ms, 25)
  // Tolerance is metadata the run does not measure, so it survives the rewrite.
  assert.equal(doc.platforms['win32-x64'].scenarios.stream.tolerance.tolFrac, 0.9)
  assert.equal(doc.platforms['win32-x64'].scenarios.keystroke.metrics.keystroke_p50_ms, 3)
})

test('a legacy file round-trips: migrate, then add a second platform', () => {
  const path = writeDoc('b.json', legacyDoc())

  updateBaseline(path, [{ name: 'stream', metrics: { frame_p95_ms: 40 } }], 'win32-x64')

  // Reading as the legacy machine still gates on the numbers it committed.
  const mac = loadBaseline(path, 'darwin-arm64')

  assert.equal(mac.gated, true)
  assert.equal(mac.scenarios.stream.metrics.frame_p95_ms, 22)
  assert.equal(loadBaseline(path, 'win32-x64').scenarios.stream.metrics.frame_p95_ms, 40)
})
