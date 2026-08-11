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

test('malformed JSON is consumed silently', () => {
  const home = tempHome()
  write(home, '{nope')

  assert.equal(readAndConsumeHandoffResult(home), null)
  assert.equal(fs.existsSync(handoffResultPath(home)), false)
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
  assert.equal(readAndConsumeHandoffResult(home)?.manual, false, 'older writers without the field parse as manual:false')
})
