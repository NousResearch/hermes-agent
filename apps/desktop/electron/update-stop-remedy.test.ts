import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test } from 'vitest'

import { explainUpdateStop, readStoppedRun, receiptDirectory } from './update-stop-remedy'

const homes: string[] = []

afterEach((): void => {
  for (const home of homes.splice(0)) {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

function home(): string {
  const dir: string = fs.mkdtempSync(path.join(os.tmpdir(), 'update-stop-'))

  homes.push(dir)

  return dir
}

function writeReceipt(root: string, record: Record<string, unknown>): void {
  const dir: string = path.join(root, 'logs', 'update_receipts')

  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, 'latest.json'), JSON.stringify(record))
}

const REFUSED = {
  schema: 1,
  update_id: 'u1',
  correlation_id: 'posix-1-2-3',
  started_at: '2026-10-19T10:00:00+00:00',
  finished_at: '2026-10-19T10:00:04+00:00',
  outcome: 'failed',
  stop_class: 'downgrade_refused',
  exit_code: 1
}

test('a failed hand-off is explained by the stop its own receipt recorded', () => {
  const root = home()

  writeReceipt(root, REFUSED)

  assert.deepEqual(readStoppedRun(root, 'posix-1-2-3'), { stopClass: 'downgrade_refused', outcome: 'failed' })
})

test('a receipt from another run never explains this one', () => {
  const root = home()

  writeReceipt(root, REFUSED)

  assert.equal(readStoppedRun(root, 'posix-9-9-9'), null)
})

test('no run identity means no attribution', () => {
  const root = home()

  writeReceipt(root, REFUSED)

  assert.equal(readStoppedRun(root, null), null)
})

test('a record still running has no stop to report', () => {
  const root = home()

  writeReceipt(root, { ...REFUSED, outcome: 'running', finished_at: null })

  assert.equal(readStoppedRun(root, 'posix-1-2-3'), null)
})

test('a run that committed is not a stop, whatever class it carries', () => {
  const root = home()

  writeReceipt(root, { ...REFUSED, outcome: 'success', stop_class: null })

  assert.equal(readStoppedRun(root, 'posix-1-2-3'), null)
})

test('a token outside the closed set is ignored', () => {
  const root = home()

  writeReceipt(root, { ...REFUSED, stop_class: 'Downgrade refused (exit 1)' })

  assert.equal(readStoppedRun(root, 'posix-1-2-3'), null)
})

test('an unreadable receipt falls back to the generic dialog', () => {
  const root = home()

  fs.mkdirSync(path.join(root, 'logs', 'update_receipts'), { recursive: true })
  fs.writeFileSync(path.join(root, 'logs', 'update_receipts', 'latest.json'), '{not json')

  assert.equal(readStoppedRun(root, 'posix-1-2-3'), null)
  assert.equal(readStoppedRun(path.join(root, 'missing'), 'posix-1-2-3'), null)
})

test('a profile home reads the root home receipts, where the update files them', () => {
  const root = home()

  writeReceipt(root, REFUSED)

  assert.equal(receiptDirectory(path.join(root, 'profiles', 'acme')), path.join(root, 'logs', 'update_receipts'))
  assert.deepEqual(readStoppedRun(path.join(root, 'profiles', 'acme'), 'posix-1-2-3'), {
    stopClass: 'downgrade_refused',
    outcome: 'failed'
  })
})

test('the downgrade refusal names the channel that unblocks it', () => {
  const explained = explainUpdateStop('downgrade_refused')

  assert.ok(explained, 'the recorded refusal has an explanation')
  assert.match(explained.message, /kept your install/)
  assert.match(explained.detail, /--set-channel main/)
  assert.match(explained.detail, /nothing was changed/)
})

test('a stop with no explanation keeps the generic failure dialog', () => {
  assert.equal(explainUpdateStop(null), null)
  assert.equal(explainUpdateStop('git_timeout'), null)
  assert.equal(explainUpdateStop('toString'), null)
})
