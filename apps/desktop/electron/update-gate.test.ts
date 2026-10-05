/**
 * Tests for electron/update-gate.ts — the update mutual-exclusion gate that
 * parks local backend spawns while an in-app update is running.
 *
 * The regression this guards (#73822): applyUpdates stops its own backend
 * before committing the hand-off. A marker-only gate lets the renderer's
 * reconnect spawn a fresh backend on the runtime being replaced.
 * The gate must consult the in-process updateInFlight flag and the successful
 * detached hand-off state as well.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { updateGateReason, waitForUpdateClearance } from './update-gate'

function deps(marker: boolean, inFlight: boolean, handoffActive = false) {
  return {
    hasLiveMarker: () => marker,
    isUpdateInFlight: () => inFlight,
    isHandoffActive: () => handoffActive
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

test('handoff remains closed after the detached wrapper exits', async () => {
  let handoffActive = true
  let ticks = 0

  const outcome = await waitForUpdateClearance(
    {
      hasLiveMarker: () => false,
      isUpdateInFlight: () => false,
      isHandoffActive: () => handoffActive
    },
    {
      onWaitTick: reason => {
        ticks += 1
        assert.equal(reason, 'handoff')

        if (ticks === 2) {
          handoffActive = false
        }
      },
      pollMs: 1,
      sleep: async () => {},
      timeoutMs: 10_000
    }
  )

  assert.equal(outcome, 'finished')
  assert.equal(ticks, 2)
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
    { hasLiveMarker: () => false, isUpdateInFlight: () => inFlight, isHandoffActive: () => false },
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
    { hasLiveMarker: () => marker, isUpdateInFlight: () => inFlight, isHandoffActive: () => false },
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
// failed-receipt signal (#122206)
// ---------------------------------------------------------------------------

test('a failed receipt outranks a live marker as the reported reason', () => {
  assert.equal(updateGateReason({ ...deps(true, false), hasFailedReceipt: () => true }), 'failed-receipt')
})

test('a failed receipt without a live marker keeps the gate open', () => {
  // The receipt only RECLASSIFIES a closed gate; it must not close an open
  // one — a failed update from last week must not defer any boot.
  assert.equal(updateGateReason({ ...deps(false, false), hasFailedReceipt: () => true }), null)
})

test('a running or partial receipt keeps the marker reason', () => {
  // Only a TERMINAL failure is actionable: "running" must keep parking.
  assert.equal(updateGateReason({ ...deps(true, false), hasFailedReceipt: () => false }), 'marker')
})

test('abandonOn returns abandoned instead of parking on a failed receipt', async () => {
  let slept = 0

  const outcome = await waitForUpdateClearance(
    { ...deps(true, false), hasFailedReceipt: () => true },
    {
      abandonOn: reason => reason === 'failed-receipt',
      pollMs: 10,
      sleep: async () => {
        slept += 1
      },
      timeoutMs: 10_000
    }
  )

  assert.equal(outcome, 'abandoned')
  assert.equal(slept, 0)
})

test('a mid-wait receipt finalization abandons the park', async () => {
  // The gate closed on a live marker (update running); the update then fails
  // and finalizes its receipt while we are parked. The wait must abandon on
  // the next poll instead of counting down to the 20-minute deadline.
  let failedReceipt = false
  let polls = 0

  const outcome = await waitForUpdateClearance(
    { ...deps(true, false), hasFailedReceipt: () => failedReceipt },
    {
      abandonOn: reason => reason === 'failed-receipt',
      onWaitTick: () => {
        polls += 1

        if (polls === 3) {
          failedReceipt = true
        }
      },
      pollMs: 1,
      sleep: async () => {},
      timeoutMs: 10_000
    }
  )

  assert.equal(outcome, 'abandoned')
  assert.equal(polls, 3)
})

test('abandonOn declining keeps the historical parking', async () => {
  let ticks = 0
  let marker = true

  const outcome = await waitForUpdateClearance(
    {
      hasLiveMarker: () => marker,
      isUpdateInFlight: () => false,
      isHandoffActive: () => false,
      hasFailedReceipt: () => true
    },
    {
      abandonOn: () => false,
      onWaitTick: () => {
        ticks += 1

        if (ticks === 2) {
          marker = false
        }
      },
      pollMs: 1,
      sleep: async () => {},
      timeoutMs: 10_000
    }
  )

  assert.equal(outcome, 'finished')
  assert.equal(ticks, 2)
})
