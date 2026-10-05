import { execFileSync } from 'node:child_process'
import { mkdtempSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { $cronRunReadOnlyVerdicts, isStoredTranscriptReadOnly } from '@/store/read-only-transcript'

import { isResumableCronRun, openCronRun, refreshCronRunWriteGate } from './open-cron-run'

// Opt in with a dependency-equipped interpreter. This is a real SQLite producer
// -> scheduler finalizer -> endpoint -> Desktop gate integration, not Electron.
// HERMES_CRON_LINEAGE_PYTHON=/path/to/python npm test -- open-cron-run-lineage.test.ts
const python = process.env.HERMES_CRON_LINEAGE_PYTHON
const root = resolve(process.cwd(), '../..')
type Row = {
  id: string
  ended_at: number | null
  last_active: number
  cron_finalized: boolean
  scheduler_owned: boolean
  source: string
}
type Receipt = { case: string; resumable: boolean; history: Row; detail: Row }

let receipts: Receipt[] = []

describe.skipIf(!python)('real compressed cron producer-to-Desktop gates', () => {
  beforeAll(() => {
    const scratch = mkdtempSync(join(process.env.HERMES_CRON_LINEAGE_SCRATCH ?? tmpdir(), 'cron-lineage-'))
    const report = join(scratch, 'receipts.xml')
    execFileSync(
      'bash',
      [
        'scripts/run_tests.sh',
        'tests/hermes_state/test_cron_compression_finalization.py',
        '-k',
        'test_scheduler_compression_projects_real_endpoint_receipts',
        '-j',
        '1',
        '--file-retries',
        '0',
        `--junitxml=${report}`
      ],
      {
        cwd: root,
        env: { ...process.env, HERMES_PYTHON: python },
        timeout: 120_000,
        encoding: 'utf8'
      }
    )
    const xml = new DOMParser().parseFromString(readFileSync(report, 'utf8'), 'application/xml')
    const receiptProperty = xml.querySelector('property[name="desktop_receipts"]')
    expect(receiptProperty).not.toBeNull()
    receipts = JSON.parse(receiptProperty!.getAttribute('value')!) as Receipt[]
    expect(receipts).toHaveLength(4)
  }, 150_000)

  beforeEach(() => $cronRunReadOnlyVerdicts.set(new Map()))

  it.each([
    'cron_complete',
    'cron_incomplete_no_output',
    'cron_complete_unfinalized',
    'cron_incomplete_no_output_unfinalized'
  ])('%s uses authoritative root history and detail at open and send', async name => {
    const receipt = receipts.find(row => row.case === name)
    expect(receipt).toBeDefined()
    const { history, detail, resumable } = receipt!
    expect(history.scheduler_owned).toBe(false)
    expect(detail.scheduler_owned).toBe(false)
    expect(isResumableCronRun(history)).toBe(resumable)
    openCronRun(history, vi.fn())
    expect(isStoredTranscriptReadOnly(history.id)).toBe(!resumable)
    expect(await refreshCronRunWriteGate(history.id, async () => detail)).toBe(!resumable)
    expect(isStoredTranscriptReadOnly(history.id)).toBe(!resumable)
    // Repeated authoritative refresh must not latch a stale verdict.
    expect(await refreshCronRunWriteGate(history.id, async () => detail)).toBe(!resumable)
  })
})
