import assert from 'node:assert/strict'

import { test } from 'vitest'

import { classifyAttachedProbeError, createAttachedLivenessTracker } from './attached-backend-liveness'

// Regression tests for the attached-monitor single-flight epoch in
// electron/main.ts (stopAttachedBackendMonitor / startAttachedBackendMonitor).
//
// main.ts cannot be imported here (Electron main), so these tests drive a
// deterministic seam that mirrors that supervision logic tick-for-tick: a
// module-level single-flight flag plus a monitor generation minted on every
// stop and on every start. The stop under test must NOT clear the in-flight
// flag; each tick captures its generation; the settling probe releases the
// flag and may declare the backend gone only while its generation is still
// current. Keep this seam in sync with main.ts; if one changes, update both.

interface PendingProbe {
  epoch: number
  resolve: () => void
  reject: (error: unknown) => void
}

function createMonitorSeam() {
  let running = false
  let probeInFlight = false
  let generation = 0
  let nextProbeId = 0
  let activeReadiness = 0
  let maxActiveReadiness = 0
  const activeByEpoch = new Map<number, number>()
  const callsByEpoch = new Map<number, number>()
  const maxActiveByEpoch = new Map<number, number>()
  const pending = new Map<number, PendingProbe>()
  const teardowns: Array<{ epoch: number; reason: string }> = []

  function stopMonitor() {
    // Mirrors stopAttachedBackendMonitor: supersede the epoch without
    // releasing a pending probe's ownership.
    generation += 1
    running = false
  }

  function startMonitor() {
    // Mirrors startAttachedBackendMonitor.
    stopMonitor()
    generation += 1
    const epoch = generation
    probeInFlight = false
    running = true
    const liveness = createAttachedLivenessTracker()

    const declareGone = (reason: string, probeGeneration: number) => {
      if (probeGeneration !== generation) {
        return
      }

      teardowns.push({ epoch, reason })
      stopMonitor()
    }

    const tick = (): number | null => {
      if (!running || probeInFlight) {
        return null
      }

      probeInFlight = true
      const tickGeneration = epoch
      const id = nextProbeId
      nextProbeId += 1
      callsByEpoch.set(tickGeneration, (callsByEpoch.get(tickGeneration) ?? 0) + 1)
      activeReadiness += 1
      maxActiveReadiness = Math.max(maxActiveReadiness, activeReadiness)
      activeByEpoch.set(tickGeneration, (activeByEpoch.get(tickGeneration) ?? 0) + 1)
      maxActiveByEpoch.set(
        tickGeneration,
        Math.max(maxActiveByEpoch.get(tickGeneration) ?? 0, activeByEpoch.get(tickGeneration) ?? 0)
      )

      let resolveProbe!: () => void
      let rejectProbe!: (error: unknown) => void
      const readiness = new Promise<void>((resolve, reject) => {
        resolveProbe = resolve
        rejectProbe = reject
      })
      pending.set(id, { epoch: tickGeneration, resolve: resolveProbe, reject: rejectProbe })

      void readiness.then(
        () => {
          pending.delete(id)
          activeReadiness -= 1
          activeByEpoch.set(tickGeneration, (activeByEpoch.get(tickGeneration) ?? 1) - 1)
          liveness.noteSuccess()

          if (tickGeneration === generation) {
            probeInFlight = false
          }
        },
        error => {
          pending.delete(id)
          activeReadiness -= 1
          activeByEpoch.set(tickGeneration, (activeByEpoch.get(tickGeneration) ?? 1) - 1)
          const kind = classifyAttachedProbeError(error)

          if (kind === 'hard') {
            liveness.noteHardFailure()
            declareGone(error instanceof Error ? error.message : String(error), tickGeneration)
          } else if (liveness.noteTransientFailure()) {
            declareGone(error instanceof Error ? error.message : String(error), tickGeneration)
          }

          if (tickGeneration === generation) {
            probeInFlight = false
          }
        }
      )

      return id
    }

    return { epoch, tick, declareGone }
  }

  return {
    startMonitor,
    stopMonitor,
    state: () => ({ running, probeInFlight, generation }),
    callsFor: (epoch: number) => callsByEpoch.get(epoch) ?? 0,
    maxActiveFor: (epoch: number) => maxActiveByEpoch.get(epoch) ?? 0,
    maxActiveReadiness: () => maxActiveReadiness,
    activeReadiness: () => activeReadiness,
    teardowns,
    settle: (id: number, error?: unknown) => {
      const probe = pending.get(id)
      assert.ok(probe, `expected pending probe ${id}`)

      if (error === undefined) {
        probe.resolve()
      } else {
        probe.reject(error)
      }
    }
  }
}

async function flushSettlements() {
  await new Promise<void>(resolve => setTimeout(resolve, 0))
  await new Promise<void>(resolve => setTimeout(resolve, 0))
}

test('steady ticks stay single-flight: a pending probe blocks the next tick', async () => {
  const seam = createMonitorSeam()
  const monitor = seam.startMonitor()
  const first = monitor.tick()

  assert.ok(first !== null)
  assert.equal(monitor.tick(), null)
  assert.equal(seam.callsFor(monitor.epoch), 1)

  seam.settle(first)
  await flushSettlements()
  assert.equal(seam.state().probeInFlight, false)

  const second = monitor.tick()

  assert.ok(second !== null)
  assert.equal(seam.callsFor(monitor.epoch), 2)
  assert.equal(seam.maxActiveReadiness(), 1)
})

test('a superseded probe success does not release the new monitor guard', async () => {
  const seam = createMonitorSeam()
  const monitorA = seam.startMonitor()
  const probeA = monitorA.tick()

  assert.ok(probeA !== null)

  // Replace the monitor while A's readiness op is still pending.
  const monitorB = seam.startMonitor()
  const probeB = monitorB.tick()

  assert.ok(probeB !== null, 'the new epoch starts with a free guard')
  assert.equal(seam.callsFor(monitorB.epoch), 1)

  // A's late success is fenced: it must not release B's guard.
  seam.settle(probeA)
  await flushSettlements()

  // B's first probe is still pending, so the next tick must not double-start.
  assert.equal(monitorB.tick(), null)
  assert.equal(seam.callsFor(monitorB.epoch), 1)
  assert.equal(seam.maxActiveFor(monitorB.epoch), 1)
  assert.equal(seam.state().probeInFlight, true)

  // The new monitor still probes once its own op settles.
  seam.settle(probeB)
  await flushSettlements()
  assert.equal(seam.state().probeInFlight, false)
  assert.ok(monitorB.tick() !== null)
  assert.equal(seam.callsFor(monitorB.epoch), 2)
})

test('a stale credentialed failure cannot tear down the new monitor slot', async () => {
  const seam = createMonitorSeam()
  const monitorA = seam.startMonitor()
  const probeA = monitorA.tick()

  assert.ok(probeA !== null)

  const monitorB = seam.startMonitor()
  const probeB = monitorB.tick()

  assert.ok(probeB !== null)

  // A credentialed 401 is a hard failure, but from a stale epoch it must be
  // fenced before it can tear down the new slot.
  seam.settle(probeA, new Error('401: no_cookie'))
  await flushSettlements()

  assert.equal(seam.teardowns.length, 0)
  assert.equal(seam.state().running, true)
  assert.equal(seam.state().probeInFlight, true)
  assert.equal(monitorB.tick(), null)
  assert.equal(seam.maxActiveFor(monitorB.epoch), 1)

  seam.settle(probeB)
  await flushSettlements()
  assert.ok(monitorB.tick() !== null)
})

test('stop retains the guard until the owner settles; the next start opens a fresh epoch', async () => {
  const seam = createMonitorSeam()
  const monitorA = seam.startMonitor()
  const probeA = monitorA.tick()

  assert.ok(probeA !== null)

  // Stopping clears the interval but must not release the pending probe.
  seam.stopMonitor()
  assert.equal(seam.state().running, false)
  assert.equal(seam.state().probeInFlight, true)

  // The superseded probe settles fenced: no release, no teardown.
  seam.settle(probeA)
  await flushSettlements()
  assert.equal(seam.state().probeInFlight, true)
  assert.equal(seam.teardowns.length, 0)

  // The next monitor starts clean on a fresh epoch.
  const monitorB = seam.startMonitor()

  assert.ok(monitorB.epoch !== monitorA.epoch)
  assert.equal(seam.state().probeInFlight, false)
  assert.ok(monitorB.tick() !== null)
})

test('a current-epoch hard failure still tears down the live slot', async () => {
  const seam = createMonitorSeam()
  const monitor = seam.startMonitor()
  const probe = monitor.tick()

  assert.ok(probe !== null)

  seam.settle(probe, new Error('401: no_cookie'))
  await flushSettlements()

  assert.equal(seam.teardowns.length, 1)
  assert.equal(seam.teardowns[0]?.epoch, monitor.epoch)
  assert.equal(seam.state().running, false)
})
