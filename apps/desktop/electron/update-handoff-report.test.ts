import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test, vi } from 'vitest'

import { handoffResultPath } from './handoff-result'
import { parkedRunReport } from './update-handoff-report'

function reportFor(home: string, park: (run: ReturnType<typeof parkedRunReport>) => void): string[] {
  const lines: string[] = []
  const run = parkedRunReport()
  park(run)
  run.report({
    hermesHome: home,
    log: line => lines.push(line),
    dialog: { showMessageBox: async () => ({ response: 2, checkboxChecked: false }) },
    shell: { showItemInFolder: () => undefined },
    openUpdates: () => undefined,
    sendReport: () => undefined
  })

  return lines
}

function writeResult(home: string, runId: string) {
  const finished_at = Math.floor(Date.now() / 1000)
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

test('a failed update offers Send report, which opens the diagnostics upload with the failure as context', async () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'handoff-report-'))
  const finished_at = Math.floor(Date.now() / 1000)
  fs.writeFileSync(
    handoffResultPath(home),
    JSON.stringify({ ok: false, exit_code: 1, message: 'uv sync failed', run_id: 'run-x', finished_at })
  )
  const sendReport = vi.fn()
  let buttons: string[] = []
  const run = parkedRunReport()
  run.park({ startedAt: null, runId: 'run-x' })
  run.report({
    hermesHome: home,
    log: () => undefined,
    dialog: {
      showMessageBox: async options => {
        buttons = options.buttons ?? []

        return { response: buttons.indexOf('Send report'), checkboxChecked: false }
      }
    },
    shell: { showItemInFolder: () => undefined },
    openUpdates: () => undefined,
    sendReport
  })

  await vi.waitFor(() => assert.equal(sendReport.mock.calls.length, 1))
  assert.ok(buttons.includes('Send report'), buttons.join(', '))
  assert.match(sendReport.mock.calls[0][0], /exit 1.*uv sync failed/)
})
