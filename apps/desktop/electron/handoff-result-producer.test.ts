import assert from 'node:assert/strict'
import { type ChildProcess, spawn } from 'node:child_process'
import { once } from 'node:events'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { setTimeout as sleep } from 'node:timers/promises'

import { test } from 'vitest'

import { handoffResultPath, readAndConsumeHandoffResult } from './handoff-result'
import { claimBridgeMarker, markerPath, parseUpdateMarker } from './update-marker'
import { liveMarkerProbe } from './update-marker-gate'

async function stop(child: ChildProcess) {
  if (child.exitCode !== null || child.signalCode !== null) {
    return
  }

  const exited = once(child, 'exit')
  child.kill('SIGKILL')
  await exited
}

test.skipIf(process.platform === 'win32').each(['legacy', 'protocol-2'])(
  '%s POSIX producer receipt survives heartbeat and a newly opened gate',
  async mode => {
    const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'handoff-correlation-'))
    const home = path.join(fixture, 'home')
    const root = path.join(fixture, 'checkout')
    const temp = path.join(fixture, 'temp')

    for (const dir of [home, temp, path.join(root, 'pm')]) {
      fs.mkdirSync(dir, { recursive: true })
    }

    // No launcher: the actual hand-off fails without invoking an update, gateway or app.
    const script = path.resolve(__dirname, '../../../scripts/desktop-update/posix.sh')
    const desktop = spawn('sleep', ['60'])
    const startedAt = Math.floor(Date.now() / 1000) - 10
    const runId = mode === 'protocol-2' ? 'receipt-test.bridge-2' : undefined
    const bridge = await claimBridgeMarker(home, { pid: desktop.pid, startedAt, runId })
    assert.equal(bridge.ok, true)

    const child = spawn(
      'bash',
      [
        script,
        '--daemonized',
        '--install-root',
        root,
        '--desktop-pid',
        String(desktop.pid),
        '--no-ui',
        '--self-test-refresh-every',
        '1',
        ...(runId ? ['--handoff-run', runId] : [])
      ],
      {
        env: {
          ...process.env,
          HOME: fixture,
          HERMES_HOME: home,
          TMPDIR: temp,
          HERMES_UPDATE_STARTED_AT: String(startedAt),
          HERMES_UPDATE_SHIM_GRACE_SECONDS: '0'
        },
        stdio: ['ignore', 'pipe', 'pipe']
      }
    )

    let output = ''
    child.stdout!.on('data', chunk => {
      output += chunk
    })
    child.stderr!.on('data', chunk => {
      output += chunk
    })
    const exited = once(child, 'exit')
    const observed: { startedAt: number | null; runId?: string | null }[] = []
    const probe = liveMarkerProbe({ hermesHome: home, reclaim: null, onLiveMarker: marker => observed.push(marker) })

    try {
      const deadline = Date.now() + 15_000

      while (Date.now() < deadline) {
        await probe()
        const marker = parseUpdateMarker(fs.readFileSync(markerPath(home), 'utf8'))

        if (marker?.pid === child.pid && marker.startedAt! > startedAt) {
          break
        }

        await sleep(50)
      }

      const refreshed = parseUpdateMarker(fs.readFileSync(markerPath(home), 'utf8'))!
      assert.equal(refreshed.pid, child.pid, output)
      assert.ok(refreshed.startedAt! > startedAt, 'the actual refresher must advance line 2')
      await probe()
      // No remembered first observation: this is a Desktop opened AFTER the refresh.
      let reopened: { startedAt: number | null; runId?: string | null } = { startedAt: null }

      const reopenedGate = liveMarkerProbe({
        hermesHome: home,
        reclaim: null,
        onLiveMarker: marker => {
          reopened = marker
        }
      })

      assert.equal(await reopenedGate(), true)
      await stop(desktop)
      const [code] = await exited
      assert.equal(code, 3, output)
      const receipt = JSON.parse(fs.readFileSync(handoffResultPath(home), 'utf8'))
      assert.equal(receipt.started_at, startedAt)

      const result = readAndConsumeHandoffResult(home, {
        expectedStartedAt: reopened.startedAt,
        expectedRunId: reopened.runId
      })

      assert.equal(result?.ok, false, `own failure must not be discarded: ${JSON.stringify(receipt)}`)
      assert.equal(result?.exitCode, 3)
      assert.match(receipt.run_id, /^[A-Za-z0-9._-]{1,128}$/)
      assert.equal(receipt.run_id, refreshed.run)
      assert.equal(reopened.runId, receipt.run_id)
      assert.equal(observed.at(-1)?.runId, receipt.run_id)

      if (runId) {
        assert.equal(receipt.run_id, runId, 'adoption preserves the Desktop identity')
      }

      assert.equal(readAndConsumeHandoffResult(home), null, 'reported once')
    } finally {
      await stop(desktop)
      await stop(child)
      fs.rmSync(fixture, { recursive: true, force: true })
    }
  },
  30_000
)
