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

test('gate open when neither marker nor flag is set', async () => {
  assert.equal(await updateGateReason(deps(false, false)), null)
})

test('marker alone closes the gate', async () => {
  assert.equal(await updateGateReason(deps(true, false)), 'marker')
})

test('updateInFlight alone closes the gate (#73822 — the pre-marker window)', async () => {
  assert.equal(await updateGateReason(deps(false, true)), 'update-in-flight')
})

test('marker wins as the reported reason when both are set', async () => {
  assert.equal(await updateGateReason(deps(true, true)), 'marker')
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

test('an in-process signal times out when it never clears', async () => {
  let clock = 0

  const outcome = await waitForUpdateClearance(deps(false, true), {
    now: () => clock,
    pollMs: 10,
    sleep: async ms => {
      clock += ms
    },
    timeoutMs: 50
  })

  assert.equal(outcome, 'timeout')
})

// A live marker owner is waited out, never aged out (C1 rule 3, desktop V3):
// the old 20-minute deadline booted a backend into a half-replaced runtime.
test('a live marker keeps parking past the deadline until its owner finishes', async () => {
  let clock = 0
  let marker = true
  let ticks = 0

  const outcome = await waitForUpdateClearance(
    { hasLiveMarker: async () => marker, isUpdateInFlight: () => false, isHandoffActive: () => false },
    {
      now: () => clock,
      onWaitTick: () => {
        ticks += 1

        if (ticks === 50) {
          marker = false
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
  assert.equal(ticks, 50, 'parked ten times past the deadline')
})

// A failed receipt never outranks a live marker (desktop V2): latest.json is
// written only at finalize, so a retry after a failed update still reads
// "failed" while the new update runs.
test('a failed receipt does not open the gate while a live marker exists', async () => {
  const withFailedReceipt = { ...deps(true, false), hasFailedReceipt: () => true }

  assert.equal(await updateGateReason(withFailedReceipt), 'marker')
})
