import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, test } from 'vitest'

import { PENDING_UPDATE_RUN_FILE, UpdateRunRecorder, type UpdateRunReport } from './update-metrics'

const dirs: string[] = []

afterEach((): void => {
  for (const dir of dirs.splice(0)) {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

function tempDir(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-update-metrics-'))
  dirs.push(dir)

  return dir
}

/** What the renderer does: claim, send, ack with the RPC's fate. */
async function send(recorder: UpdateRunRecorder, rpc: (run: UpdateRunReport) => Promise<unknown>): Promise<boolean> {
  const run = recorder.take()

  if (!run) {
    return false
  }

  let sent = false

  try {
    await rpc(run)
    sent = true
  } catch {
    // The record must survive for the next attach.
  } finally {
    recorder.ack(sent)
  }

  return sent
}

test('a hand-off run is reported once as success after the restart, and deleted only once the RPC resolves', async (): Promise<void> => {
  const dir = tempDir()
  const file = path.join(dir, PENDING_UPDATE_RUN_FILE)
  let now = 1_000_000

  const before = new UpdateRunRecorder({ dir: () => dir, appVersion: () => '1.0.0', now: () => now })

  await before.track('electron-updater', 1_700_000_000, async () => ({ ok: true, handedOff: true }))
  // Same process after the hand-off: the installer owns the outcome now.
  expect(before.take()).toBeNull()
  expect(fs.existsSync(file)).toBe(true)

  now += 45_000
  const after = new UpdateRunRecorder({ dir: () => dir, appVersion: () => '1.1.0', now: () => now })
  const sent: UpdateRunReport[] = []

  expect(await send(after, () => Promise.reject(new Error('backend gone')))).toBe(false)
  expect(fs.existsSync(file)).toBe(true)

  const claimed = after.take()
  expect(claimed).not.toBeNull()
  // A second caller while the first claim is unacked sends nothing.
  expect(await send(after, async run => sent.push(run))).toBe(false)
  after.ack(true)

  expect(fs.existsSync(file)).toBe(false)
  expect(await send(after, async run => sent.push(run))).toBe(false)
  expect(claimed).toEqual({
    outcome: 'success',
    mechanism: 'electron-updater',
    duration_ms: 45_000,
    from_commit_date: 1_700_000_000
  })
})

test('a failed run reports only a bucketed stage, never error text', async (): Promise<void> => {
  const allowed = new Set(['download', 'verify', 'apply', 'restart', 'other'])
  const secret = 'ENOENT /Users/alice/Hermes.app https://feed.example/v9.9.9'

  for (const stages of [[], ['fetch'], ['fetch', 'prepare'], ['fetch', 'prepare', 'restart', 'error'], ['weird']]) {
    const dir = tempDir()
    const recorder = new UpdateRunRecorder({ dir: () => dir, appVersion: () => '1.0.0' })

    await expect(
      recorder.track('microsoft-store', null, async () => {
        stages.forEach((stage: string): void => recorder.noteProgress(stage))
        throw new Error(secret)
      })
    ).rejects.toThrow(secret)

    const report = recorder.take()
    expect(report?.outcome).toBe('failed')
    expect(allowed.has(String(report?.failed_stage))).toBe(true)
    expect(JSON.stringify(report)).not.toMatch(/alice|feed\.example|9\.9\.9|ENOENT/)
    expect(fs.readFileSync(path.join(dir, PENDING_UPDATE_RUN_FILE), 'utf8')).not.toMatch(/alice|ENOENT/)
  }
})
