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

test('stable run identity outranks heartbeat time; only legacy receipts use timestamp correlation', () => {
  const home = tempHome()
  const finished_at = Math.floor(Date.now() / 1000)

  const outcomes = [
    { ok: false, exit_code: 3, message: 'failed' },
    { ok: true, exit_code: 0, manual: true, message: 'reopen manually' },
    { ok: true, exit_code: 0, warnings: ['gateway restart'] }
  ]

  for (const outcome of outcomes) {
    const started_at = finished_at - 600
    const receipt = { ...outcome, started_at, finished_at }
    write(home, { ...receipt, run_id: 'own-run' })
    assert.ok(readAndConsumeHandoffResult(home, { expectedRunId: 'own-run', expectedStartedAt: finished_at + 60 }))

    for (const run_id of ['foreign-run', '', null, 42, 'bad run', 'own-run\n']) {
      write(home, { ...receipt, run_id })
      assert.equal(readAndConsumeHandoffResult(home, { expectedRunId: 'own-run', expectedStartedAt: started_at }), null)
    }

    // An older script has no run_id, even if its marker carried a run: line.
    write(home, receipt)
    assert.ok(readAndConsumeHandoffResult(home, { expectedRunId: 'own-run', expectedStartedAt: started_at }))
    // ...and line 2 from a run that began after this result finished is not this run.
    write(home, receipt)
    assert.equal(readAndConsumeHandoffResult(home, { expectedRunId: 'own-run', expectedStartedAt: finished_at + 60 }), null)
    // Older markers and post-update boots have no expected stable identity.
    write(home, { ...receipt, run_id: 'own-run' })
    assert.ok(readAndConsumeHandoffResult(home, { expectedStartedAt: started_at }))
    write(home, { ...receipt, run_id: 'own-run' })
    assert.ok(readAndConsumeHandoffResult(home))
  }

  fs.rmSync(home, { recursive: true, force: true })
})

// Legacy producers correlate started_at; a result from another run is not
// reported as the one this boot waited on.
test('a result from a run that finished before the parked one is discarded; a match is reported with warnings', () => {
  const home = tempHome()
  const now = Math.floor(Date.now() / 1000)
  const parked = now - 300
  write(home, { ok: false, exit_code: 1, message: 'old run', branch: 'main', started_at: now - 900, finished_at: now - 800 })
  assert.equal(readAndConsumeHandoffResult(home, { expectedStartedAt: parked }), null)

  write(home, { ok: true, exit_code: 0, message: '', branch: 'main', started_at: parked, finished_at: now, warnings: ['gateway restart'] })
  assert.deepEqual(readAndConsumeHandoffResult(home, { expectedStartedAt: parked })?.warnings, ['gateway restart'])
})

// Review 5411223284 regression 2: with no `run:` line the gate sees line 2,
// which the scripts' 300 s heartbeat moves while their run lives. A run
// longer than 5 minutes must still have its result reported.
test('line 2 moved by the heartbeat inside the run still correlates its result', () => {
  const home = tempHome()
  const now = Math.floor(Date.now() / 1000)
  const started_at = now - 900
  const result = { ok: false, exit_code: 3, message: 'failed after 15 min', branch: 'main', started_at, finished_at: now }

  for (const heartbeat of [started_at + 300, started_at + 600, now]) {
    write(home, { ...result, run_id: 'script-run' })
    assert.equal(readAndConsumeHandoffResult(home, { expectedStartedAt: heartbeat })?.message, 'failed after 15 min')
    write(home, result)
    assert.equal(readAndConsumeHandoffResult(home, { expectedStartedAt: heartbeat })?.exitCode, 3)
  }

  // Line 2 from before this run started, or after it finished, is another run.
  for (const other of [started_at - 1, now + 1]) {
    write(home, result)
    assert.equal(readAndConsumeHandoffResult(home, { expectedStartedAt: other }), null)
  }

  fs.rmSync(home, { recursive: true, force: true })
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
