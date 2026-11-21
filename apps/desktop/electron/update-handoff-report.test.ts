import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import type { MessageBoxOptions } from 'electron'
import { test } from 'vitest'

import { handoffResultPath } from './handoff-result'
import { parkedRunReport } from './update-handoff-report'

function reportFor(
  home: string,
  park: (run: ReturnType<typeof parkedRunReport>) => void,
  dialogs: MessageBoxOptions[] = []
): string[] {
  const lines: string[] = []
  const run = parkedRunReport()
  park(run)
  run.report({
    hermesHome: home,
    log: line => lines.push(line),
    dialog: {
      showMessageBox: async options => {
        dialogs.push(options)

        return { response: 2, checkboxChecked: false }
      }
    },
    shell: { showItemInFolder: () => undefined },
    openUpdates: () => undefined
  })

  return lines
}

function writeResult(home: string, runId: string, finished_at = Math.floor(Date.now() / 1000)) {
  fs.writeFileSync(
    handoffResultPath(home),
    JSON.stringify({
      ok: true,
      exit_code: 0,
      message: 'done',
      branch: 'main',
      run_id: runId,
      started_at: finished_at - 60,
      finished_at
    })
  )
}

test('the boot wait reports the result of the run it last parked on, across line-2 heartbeats', () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'handoff-report-'))
  const now = Math.floor(Date.now() / 1000)

  writeResult(home, 'run-b')

  const heartbeat = reportFor(home, run => {
    run.park({ startedAt: now - 600, runId: 'run-b' })
    run.park({ startedAt: now, runId: 'run-b' }) // the script refreshed line 2 mid-wait
  })

  assert.ok(
    heartbeat.some(line => line.includes('finished OK')),
    heartbeat.join('\n')
  )

  writeResult(home, 'run-a')
  const foreign = reportFor(home, run => run.park({ startedAt: now, runId: 'run-b' }))
  assert.ok(
    foreign.some(line => line.includes('for run run-a, not run-b; discarded')),
    foreign.join('\n')
  )
  assert.equal(fs.existsSync(handoffResultPath(home)), false, 'a foreign result is still consumed')
})

test('a completed update shows its own gateway migration blocker even on a later launch', () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'handoff-report-'))
  const dir = path.join(home, 'logs', 'update_receipts')
  const blocker = 'Profiles default and example share a Telegram bot token. Resolve the duplicate, then run hermes gateway migrate --multiplex.'
  const finishedAt = Math.floor(Date.now() / 1000) - 3600

  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(
    path.join(dir, 'latest.json'),
    JSON.stringify({
      correlation_id: 'run-blocked',
      finished_at: new Date(finishedAt * 1000).toISOString(),
      outcome: 'success',
      followups: [{ step: 'gateway_migration', reason: blocker }]
    })
  )

  const dialogs: MessageBoxOptions[] = []
  writeResult(home, 'run-other')
  reportFor(home, () => {}, dialogs)
  assert.equal(dialogs.length, 0, 'a different update must not inherit the blocker')

  writeResult(home, 'run-blocked', finishedAt)
  reportFor(home, () => {}, dialogs)
  assert.equal(dialogs.length, 1)
  assert.equal(dialogs[0].type, 'warning')
  assert.equal(dialogs[0].detail, blocker)

  reportFor(home, () => {}, dialogs)
  assert.equal(dialogs.length, 1, 'the result is consumed once')
})

function failedReport(home: string, runId: string, stopClass: string | null): { lines: string[]; dialogs: any[] } {
  const lines: string[] = []
  const dialogs: any[] = []
  const finished_at = Math.floor(Date.now() / 1000)

  fs.writeFileSync(
    handoffResultPath(home),
    JSON.stringify({
      ok: false,
      exit_code: 1,
      message: 'Update failed (exit 1). Run hermes debug share in a terminal to send a report.',
      branch: 'main',
      run_id: runId,
      started_at: finished_at - 60,
      finished_at
    })
  )

  if (stopClass !== null) {
    const dir = path.join(home, 'logs', 'update_receipts')

    fs.mkdirSync(dir, { recursive: true })
    fs.writeFileSync(
      path.join(dir, 'latest.json'),
      JSON.stringify({
        schema: 1,
        update_id: 'u1',
        correlation_id: runId,
        started_at: '2026-10-19T10:00:00+00:00',
        finished_at: '2026-10-19T10:00:04+00:00',
        outcome: 'failed',
        stop_class: stopClass,
        exit_code: 1
      })
    )
  }

  const run = parkedRunReport()

  run.park({ startedAt: finished_at - 60, runId })
  run.report({
    hermesHome: home,
    log: line => lines.push(line),
    dialog: {
      showMessageBox: async (options: any) => {
        dialogs.push(options)

        return { response: 2, checkboxChecked: false }
      }
    },
    shell: { showItemInFolder: () => undefined },
    openUpdates: () => undefined
  })

  return { lines, dialogs }
}

test('a failed run the receipt explains names its stop instead of the bare exit code', () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'handoff-report-'))
  const { lines, dialogs } = failedReport(home, 'run-refused', 'downgrade_refused')

  assert.equal(dialogs.length, 1, 'a failed hand-off gets one dialog')
  assert.match(dialogs[0].message, /kept your install/)
  assert.match(dialogs[0].detail, /--set-channel main/)
  assert.match(dialogs[0].detail, /Details: Update failed \(exit 1\)/, 'the raw message stays visible')
  assert.deepEqual(dialogs[0].buttons, ['Update settings', 'Open log', 'Close'], 'a refusal is not retried')
  assert.ok(
    lines.some(line => line.includes('detached update FAILED (exit 1)') && line.includes('stop_class downgrade_refused')),
    lines.join('\n')
  )
})

test('a failed run with no receipt keeps the generic retry dialog', () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'handoff-report-'))
  const { lines, dialogs } = failedReport(home, 'run-plain', null)

  assert.equal(dialogs.length, 1)
  assert.equal(dialogs[0].message, "Hermes couldn't finish updating")
  assert.deepEqual(dialogs[0].buttons, ['Try again', 'Open log', 'Close'])
  assert.ok(
    lines.some(line => line.includes('detached update FAILED (exit 1)') && !line.includes('stop_class')),
    lines.join('\n')
  )
})
