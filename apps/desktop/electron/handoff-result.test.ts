import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { handoffResultPath, readAndConsumeHandoffResult } from './handoff-result'

function tempHome(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'handoff-result-'))
}

function write(home: string, body: any) {
  fs.writeFileSync(handoffResultPath(home), typeof body === 'string' ? body : JSON.stringify(body))
}

test('consumes and returns a fresh failure result', () => {
  const home = tempHome()
  write(home, {
    ok: false,
    exit_code: 6,
    message: 'rebuild failed',
    branch: 'main',
    finished_at: Math.floor(Date.now() / 1000)
  })

  const result = readAndConsumeHandoffResult(home)

  assert.ok(result)
  assert.equal(result.ok, false)
  assert.equal(result.exitCode, 6)
  assert.equal(result.message, 'rebuild failed')
  assert.equal(fs.existsSync(handoffResultPath(home)), false, 'result file must be consumed')
})

test('reports each result at most once', () => {
  const home = tempHome()
  write(home, { ok: true, exit_code: 0, message: 'done', branch: 'main', finished_at: Math.floor(Date.now() / 1000) })

  assert.ok(readAndConsumeHandoffResult(home))
  assert.equal(readAndConsumeHandoffResult(home), null)
})

test('discards stale results but still consumes the file', () => {
  const home = tempHome()
  write(home, {
    ok: false,
    exit_code: 5,
    message: 'old',
    branch: 'main',
    finished_at: Math.floor(Date.now() / 1000) - 3600
  })

  assert.equal(readAndConsumeHandoffResult(home), null)
  assert.equal(fs.existsSync(handoffResultPath(home)), false)
})

// V19: a torn/malformed result is never dropped silently — it is logged and
// kept as `.corrupt` (it used to be unlinked before parsing).
test('malformed JSON is logged and kept as .corrupt, not reported', () => {
  const home = tempHome()
  const logs: string[] = []
  write(home, '{nope')

  assert.equal(readAndConsumeHandoffResult(home, { log: line => logs.push(line) }), null)
  assert.equal(fs.existsSync(handoffResultPath(home)), false)
  assert.equal(fs.readFileSync(`${handoffResultPath(home)}.corrupt`, 'utf8'), '{nope')
  assert.match(logs.join('\n'), /not valid JSON/)
})

// C2: the result carries marker line 2 as started_at; a result from another
// run is not reported as the one this boot waited on.
test('a result whose started_at differs from the parked run is discarded; a match is reported with warnings', () => {
  const home = tempHome()
  const finished_at = Math.floor(Date.now() / 1000)
  write(home, { ok: false, exit_code: 1, message: 'old run', branch: 'main', started_at: 100, finished_at })
  assert.equal(readAndConsumeHandoffResult(home, { expectedStartedAt: 200 }), null)

  write(home, { ok: true, exit_code: 0, message: '', branch: 'main', started_at: 200, finished_at, warnings: ['gateway restart'] })
  assert.deepEqual(readAndConsumeHandoffResult(home, { expectedStartedAt: 200 })?.warnings, ['gateway restart'])
})

test('absent file returns null', () => {
  assert.equal(readAndConsumeHandoffResult(tempHome()), null)
})

test('manual flag survives the round trip and defaults false', () => {
  const home = tempHome()
  write(home, {
    ok: true,
    exit_code: 0,
    manual: true,
    message: 'Update complete. Reopen Hermes to finish (it could not restart itself).',
    branch: 'main',
    finished_at: Math.floor(Date.now() / 1000)
  })

  const result = readAndConsumeHandoffResult(home)

  assert.ok(result)
  assert.equal(result.ok, true)
  assert.equal(result.manual, true)

  write(home, { ok: true, exit_code: 0, message: 'done', branch: 'main', finished_at: Math.floor(Date.now() / 1000) })
  assert.equal(
    readAndConsumeHandoffResult(home)?.manual,
    false,
    'older writers without the field parse as manual:false'
  )
})

test('an old manual result survives the freshness window but an old ordinary one does not', () => {
  const stale = Math.floor(Date.now() / 1000) - 3600

  const ordinary = tempHome()
  write(ordinary, { ok: true, exit_code: 0, manual: false, message: 'done', branch: 'main', finished_at: stale })
  assert.equal(readAndConsumeHandoffResult(ordinary), null, 'a stale ordinary result is discarded')
  assert.equal(fs.existsSync(handoffResultPath(ordinary)), false, 'and still consumed')

  const home = tempHome()
  write(home, {
    ok: true,
    exit_code: 0,
    manual: true,
    message: 'Update complete. Reopen Hermes to finish (it could not restart itself).',
    branch: 'main',
    finished_at: stale
  })

  const result = readAndConsumeHandoffResult(home)

  assert.ok(result, 'a stale manual result is still surfaced — it is the last-resort channel')
  assert.equal(result.manual, true)
  assert.equal(readAndConsumeHandoffResult(home), null, 'but only once')
})
