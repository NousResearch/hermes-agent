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
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { updateGateReason, waitForLocalBackendClearance, waitForUpdateClearance } from './update-gate'
import { UPDATE_MARKER_MAX_AGE_MS } from './update-marker'

function deps(marker: boolean, inFlight: boolean) {
  return {
    hasLiveMarker: () => marker,
    isUpdateInFlight: () => inFlight
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

test('marker wins as the reported reason when both are set', () => {
  assert.equal(updateGateReason(deps(true, true)), 'marker')
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
    { hasLiveMarker: () => false, isUpdateInFlight: () => inFlight },
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
    { hasLiveMarker: () => marker, isUpdateInFlight: () => inFlight },
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
// waitForLocalBackendClearance — bounded park (2026-09-06 review, SUB P0-1)
//
// The loop was `while (true)`: an unreadable, malformed, future-dated or
// cleanup-race marker that nothing could self-heal parked the backend forever.
// ---------------------------------------------------------------------------

test('local backend park gives up with timeout once the blocked budget is spent', async () => {
  let clock = 0
  const stillBlocked: string[] = []

  const outcome = await waitForLocalBackendClearance(deps(true, false), {
    blockedBudgetMs: 120,
    now: () => clock,
    onStillBlocked: reason => {
      stillBlocked.push(reason)
    },
    pollMs: 10,
    sleep: async ms => {
      clock += ms
    },
    timeoutMs: 50
  })

  assert.equal(outcome, 'timeout')
  // Three 50 ms windows (150 ms) cover the 120 ms budget; each window reports.
  assert.deepEqual(stillBlocked, ['marker', 'marker', 'marker'])
  assert.ok(clock >= 120 && clock < 200, `parked ${clock}ms`)
})

test('local backend park defaults its budget to the marker age ceiling', async () => {
  let clock = 0
  let windows = 0

  const outcome = await waitForLocalBackendClearance(deps(true, false), {
    now: () => clock,
    onStillBlocked: () => {
      windows += 1
    },
    pollMs: 1_000,
    sleep: async ms => {
      clock += ms
    },
    timeoutMs: UPDATE_MARKER_MAX_AGE_MS
  })

  assert.equal(outcome, 'timeout')
  assert.equal(windows, 1, 'one full window equals the default budget')
  assert.equal(clock, UPDATE_MARKER_MAX_AGE_MS)
})

test('local backend park still finishes when the gate opens inside the budget', async () => {
  let clock = 0
  let marker = true
  let windows = 0

  const outcome = await waitForLocalBackendClearance(
    { hasLiveMarker: () => marker, isUpdateInFlight: () => false },
    {
      blockedBudgetMs: 10_000,
      now: () => clock,
      onStillBlocked: () => {
        windows += 1

        if (windows === 2) {
          marker = false // the updater finished during the third window
        }
      },
      pollMs: 10,
      sleep: async ms => {
        clock += ms
      },
      timeoutMs: 50
    }
  )

  assert.equal(outcome, 'finished')
  assert.equal(windows, 2)
})
