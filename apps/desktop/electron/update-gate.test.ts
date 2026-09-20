/**
 * Tests for electron/update-gate.ts — the update mutual-exclusion gate that
 * parks local backend spawns while an in-app update is running.
 *
 * The regression this guards (#73822): applyUpdates kills its own backend
 * BEFORE the Windows venv-blocker scan but writes the on-disk marker AFTER
 * it. A marker-only gate therefore let the renderer's reconnect spawn a
 * fresh backend inside the update's own critical section, which the scan
 * reported as a blocker — aborting every Desktop update attempt on Windows.
 * The gate must consult the in-process updateInFlight flag as well.
 *
 * The second regression (#116375, Windows): the marker pre-write names the
 * short-lived `cmd start` wrapper and updateInFlight clears ~50ms after the
 * hand-off launches, so BOTH signals went false for the ~1-2s before the
 * hand-off script claims the marker. The gate reported 'finished', the
 * desktop respawned a backend, and that backend kept the Electron main
 * process alive past the script's "wait for the Desktop to exit" step — which
 * then aborted the hand-off after 30s with exit code 4 ("the Hermes window
 * (pid …) did not exit within 30s. Nothing was changed."). The gate must
 * consult the process's quit-for-handoff latch across that gap.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test } from 'vitest'

import { updateGateReason, waitForUpdateClearance } from './update-gate'

function deps(marker: boolean, inFlight: boolean, handoff = false) {
  return {
    hasLiveMarker: () => marker,
    isUpdateInFlight: () => inFlight,
    isHandoffLaunched: () => handoff
  }
}

// ---------------------------------------------------------------------------
// updateGateReason
// ---------------------------------------------------------------------------

test('gate open when neither marker nor flag is set', () => {
  assert.equal(updateGateReason(deps(false, false)), null)
})

test('marker alone closes the gate', () => {
  assert.equal(updateGateReason(deps(true, false)), 'marker')
})

test('updateInFlight alone closes the gate (#73822 — the pre-marker window)', () => {
  assert.equal(updateGateReason(deps(false, true)), 'update-in-flight')
})

test('a launched hand-off alone closes the gate (#116375 — the wrapper gap)', () => {
  // Neither a live marker nor updateInFlight is left by the time the
  // `cmd start` wrapper has exited; without this signal the gate reopens
  // mid-hand-off and the desktop respawns the backend it just released.
  assert.equal(updateGateReason(deps(false, false, true)), 'handoff')
})

test('marker wins as the reported reason when both are set', () => {
  assert.equal(updateGateReason(deps(true, true)), 'marker')
})

test('marker wins as the reported reason over a launched hand-off', () => {
  assert.equal(updateGateReason(deps(true, false, true)), 'marker')
})

test('updateInFlight wins as the reported reason over a launched hand-off', () => {
  assert.equal(updateGateReason(deps(false, true, true)), 'update-in-flight')
})

// ---------------------------------------------------------------------------
// waitForUpdateClearance
// ---------------------------------------------------------------------------

test('returns clear immediately without sleeping when the gate is open', async () => {
  let slept = 0

  const outcome = await waitForUpdateClearance(deps(false, false), {
    pollMs: 10,
    sleep: async () => {
      slept += 1
    },
    timeoutMs: 1000
  })

  assert.equal(outcome, 'clear')
  assert.equal(slept, 0)
})

test('parks on the in-flight flag and finishes when it clears', async () => {
  // Simulates the #73822 sequence: the reconnect arrives while updateInFlight
  // is true and no marker exists yet; the flag clears (abort path finally)
  // and the waiter proceeds.
  let inFlight = true
  let ticks = 0

  const outcome = await waitForUpdateClearance(
    { hasLiveMarker: () => false, isUpdateInFlight: () => inFlight, isHandoffLaunched: () => false },
    {
      onWaitTick: reason => {
        ticks += 1
        assert.equal(reason, 'update-in-flight')

        if (ticks >= 3) {
          inFlight = false
        }
      },
      pollMs: 1,
      sleep: async () => {},
      timeoutMs: 10_000
    }
  )

  assert.equal(outcome, 'finished')
  assert.equal(ticks, 3)
})

test('parks across the flag→marker handoff without a gap', async () => {
  // Success path: the marker is written (main.ts:2936) BEFORE applyUpdates'
  // finally clears the flag, so a waiter that arrived during the scan stays
  // parked through the transition instead of slipping through.
  let inFlight = true
  let marker = false
  let ticks = 0
  const reasons: string[] = []

  const outcome = await waitForUpdateClearance(
    { hasLiveMarker: () => marker, isUpdateInFlight: () => inFlight, isHandoffLaunched: () => false },
    {
      onWaitTick: reason => {
        ticks += 1
        reasons.push(reason)

        if (ticks === 2) {
          marker = true // updater hand-off: marker written first…
        }

        if (ticks === 3) {
          inFlight = false // …then the flag clears; marker still holds the gate
        }

        if (ticks === 5) {
          marker = false // updater finished
        }
      },
      pollMs: 1,
      sleep: async () => {},
      timeoutMs: 10_000
    }
  )

  assert.equal(outcome, 'finished')
  assert.deepEqual(reasons, ['update-in-flight', 'update-in-flight', 'marker', 'marker', 'marker'])
})

test('parks across the Windows wrapper gap instead of reporting the update finished', async () => {
  // The #116375 shape, to the millisecond resolution of the report: the
  // detached child is the `cmd start` WRAPPER that exits 0 immediately, so
  // observeUpdaterHandoff settles ok, the quit-for-handoff latch lands and
  // applyUpdates' finally clears updateInFlight — while
  // scripts/desktop-update/windows.ps1 only claims the marker ~1.8s later.
  // Both old signals are false in that gap; the waiter must NOT report
  // 'finished' (which let the desktop spawn a backend, keep itself alive, and
  // trip the script's 30s "did not exit" abort with exit 4). With the gate
  // held shut the waiter parks until the quit that the hand-off scheduled
  // seals local startup and cancels it.
  let inFlight = true
  let handoff = false
  let cancelled = false
  let ticks = 0
  const reasons: string[] = []

  const outcome = await waitForUpdateClearance(
    { hasLiveMarker: () => false, isUpdateInFlight: () => inFlight, isHandoffLaunched: () => handoff },
    {
      isCancelled: () => cancelled,
      onWaitTick: reason => {
        ticks += 1
        reasons.push(reason)

        if (ticks === 1) {
          // ~50ms in: the wrapper has exited 0, so observeUpdaterHandoff
          // settles ok, the quit-for-handoff latch lands and the finally
          // clears updateInFlight. The placeholder marker already reads stale.
          handoff = true
          inFlight = false
        }

        if (ticks === 4) {
          // ~2.4s in: the dwell ends and the desktop quits for the hand-off,
          // sealing local backend startup and cancelling this waiter.
          cancelled = true
        }
      },
      pollMs: 1,
      sleep: async () => {},
      timeoutMs: 10_000
    }
  )

  assert.deepEqual(reasons, ['update-in-flight', 'handoff', 'handoff', 'handoff'])
  assert.equal(outcome, 'cancelled')
})

test('returns timeout when the gate never opens', async () => {
  let clock = 0

  const outcome = await waitForUpdateClearance(deps(true, false), {
    now: () => clock,
    pollMs: 10,
    sleep: async ms => {
      clock += ms
    },
    timeoutMs: 50
  })

  assert.equal(outcome, 'timeout')
})

// ---------------------------------------------------------------------------
// main.ts wiring
// ---------------------------------------------------------------------------

// main.ts has no module.exports, so the wiring of the extracted gate into the
// main process follows the repo's source-assertion pattern (see
// hardening.test.ts). The three deps are the whole gate: dropping one is a
// silent regression in update-gate.ts's own contract, not a type error.
const GATE_TEST_DIR = path.dirname(fileURLToPath(import.meta.url))

test('main.ts wires all three gate signals into updateGateDeps', () => {
  const source = fs.readFileSync(path.join(GATE_TEST_DIR, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')
  const fnStart = source.indexOf('function updateGateDeps()')

  assert.notEqual(fnStart, -1, 'updateGateDeps must exist in main.ts')

  const fnEnd = source.indexOf('\nfunction ', fnStart + 1)
  const body = source.slice(fnStart, fnEnd === -1 ? undefined : fnEnd)

  assert.match(body, /hasLiveMarker: \(\) => Boolean\(readLiveUpdateMarker\(HERMES_HOME\)\)/)
  assert.match(body, /isUpdateInFlight: \(\) => updateInFlight\b/)
  assert.match(
    body,
    /isHandoffLaunched: \(\) => isQuittingForHandoff\b/,
    'the launched-hand-off signal must be the process quit-for-handoff latch'
  )
})
