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

import { hostRendezvousDirectory } from './host-published-token'
import { updateGateReason, waitForUpdateClearance } from './update-gate'
import { HOST_UPDATE_RESTART_RECORD, hostFleetRestartPending } from './update-gate-host-obligation'

function deps(marker: boolean, inFlight: boolean, handoffActive = false, fleetRestartPending = false) {
  return {
    hasLiveMarker: () => marker,
    isUpdateInFlight: () => inFlight,
    isHandoffActive: () => handoffActive,
    hasFleetRestartPending: () => fleetRestartPending
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
      isHandoffActive: () => handoffActive,
      hasFleetRestartPending: () => false
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
    { hasLiveMarker: () => false, isUpdateInFlight: () => inFlight, isHandoffActive: () => false, hasFleetRestartPending: () => false },
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
    { hasLiveMarker: () => marker, isUpdateInFlight: () => inFlight, isHandoffActive: () => false, hasFleetRestartPending: () => false },
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
      hasFailedReceipt: () => true,
      hasFleetRestartPending: () => false
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

// ---------------------------------------------------------------------------
// host fleet-restart obligation (#126177) — the external-update signal
// ---------------------------------------------------------------------------

test('fleet-restart-pending alone closes the gate (#126177 — external update, marker released)', () => {
  // The reported window: an update driven from OUTSIDE this process has
  // released its live-update marker, this process's flags are all false, and
  // the fleet restart is still in flight. A gate without the host-obligation
  // signal let the backend respawn race the bouncing gateway.
  assert.equal(updateGateReason(deps(false, false, false, true)), 'fleet-restart-pending')
})

test('parks across the marker→fleet-restart handoff until verification discharges it', async () => {
  // Sequence of an external `hermes update`: the live marker holds the gate
  // while the code swaps, then the updater releases it and enters its
  // fleet-restart tail. The host obligation — armed before the pull and
  // discharged only after fleet verification — must carry the gate across
  // that transition instead of letting a respawn slip into the gap.
  let marker = true
  let fleetPending = true
  let ticks = 0
  const reasons: string[] = []

  const outcome = await waitForUpdateClearance(
    {
      hasLiveMarker: () => marker,
      isUpdateInFlight: () => false,
      isHandoffActive: () => false,
      hasFleetRestartPending: () => fleetPending
    },
    {
      onWaitTick: reason => {
        ticks += 1
        reasons.push(reason)

        if (ticks === 2) {
          marker = false // updater released the live marker; restart tail begins
        }

        if (ticks === 5) {
          fleetPending = false // fleet verification discharged the obligation
        }
      },
      pollMs: 1,
      sleep: async () => {},
      timeoutMs: 10_000
    }
  )

  assert.equal(outcome, 'finished')
  assert.deepEqual(reasons, ['marker', 'marker', 'fleet-restart-pending', 'fleet-restart-pending', 'fleet-restart-pending'])
})

const HOST_ENV = { home: '/home/tester', platform: 'darwin' }

function hostIo(files: Record<string, string>, existing: Set<string> = new Set()) {
  return {
    exists: (target: string) => existing.has(target) || target in files,
    readFile: (target: string) => {
      if (target in files) {
        return files[target]
      }

      throw new Error('ENOENT')
    }
  }
}

const hostRecordPath = () =>
  `${hostRendezvousDirectory(HOST_ENV)}/${HOST_UPDATE_RESTART_RECORD}`

test('host obligation record: absent means nothing owed', () => {
  assert.equal(hostFleetRestartPending(HOST_ENV, hostIo({})), false)
})

test('host obligation record: present without a restart proof is still owed', () => {
  // The exact record hermes update writes before pulling (version, sha…):
  // no `restarted` field yet, so the restart is in flight or still to come.
  const record = JSON.stringify({ version: 1, started: 1, pid: 7, expected_sha: 'a'.repeat(40) })
  assert.equal(
    hostFleetRestartPending(HOST_ENV, hostIo({ [hostRecordPath()]: record })),
    true
  )
})

test('host obligation record: a recorded restart proof discharges the obligation', () => {
  // _verify_fleet_after_update discharges via mark_host_restart_completed:
  // the record gains a `restarted` object and stays on disk as proof for
  // other profiles' CLIs — the gate must read that as "not pending".
  const record = JSON.stringify({
    version: 1,
    started: 1,
    pid: 7,
    expected_sha: 'a'.repeat(40),
    restarted: { sha: 'a'.repeat(40), pid: 8, at: 2 }
  })

  assert.equal(
    hostFleetRestartPending(HOST_ENV, hostIo({ [hostRecordPath()]: record })),
    false
  )
})

test('host obligation record: corrupt or unreadable terms fail closed', () => {
  // Same contract as the Python reader (host_obligation_present): a record
  // whose terms cannot be read is an obligation still owed, never discharged.
  assert.equal(
    hostFleetRestartPending(HOST_ENV, hostIo({ [hostRecordPath()]: 'not json' })),
    true
  )
  // Present on disk but unreadable (permissions): still owed.
  assert.equal(
    hostFleetRestartPending(HOST_ENV, hostIo({}, new Set([hostRecordPath()]))),
    true
  )
})
