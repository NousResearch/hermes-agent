/**
 * Hand-off acknowledgment (R5, A7 rule 6, SPEC 4a.5 / 4b) against REAL
 * processes: the Desktop counts a hand-off as taken only when the marker's
 * line-1 owner is a LIVE incarnation that is not the Desktop and that is
 * correlated with THIS hand-off — protocol 2 by the `run:` id it passed (and a
 * verified creation time), legacy scripts by the started_at they echo.
 * A claimant that died before the confirmation is never a running update.
 */

import fs from 'fs'
import assert from 'node:assert/strict'

import { afterEach, describe, test } from 'vitest'

import { claimBridgeMarker, markerPath, processCreateTime, waitForHandoffClaim } from './update-marker'
import {
  claimantChild,
  cleanupMarkerFixtures,
  deadPid,
  killAndReap,
  nowSeconds,
  tmpHome
} from './update-marker.test-helpers'

const HAS_CT_PROBE = process.platform === 'linux' || process.platform === 'darwin' || process.platform === 'win32'
const RUN = 'desk-4242-mabc123-0f0f'

afterEach(cleanupMarkerFixtures)

describe.skipIf(!HAS_CT_PROBE)('protocol 2 acknowledgment (R5)', () => {
  test('a claimant that published our run and was SIGKILLed before the ack is NOT a hand-off', async () => {
    const home = tmpHome('ack-dead')
    const startedAt = nowSeconds()
    await claimBridgeMarker(home, { startedAt, runId: RUN })
    const { child, body } = await claimantChild(markerPath(home), { startedAt, run: RUN })

    await killAndReap(child)

    assert.equal(fs.readFileSync(markerPath(home), 'utf8'), body, 'the dead claim is still on disk')
    assert.deepEqual(await waitForHandoffClaim(home, process.pid, { runId: RUN, timeoutMs: 300, pollMs: 50 }), {
      taken: false
    })
  })

  test('a live claimant carrying a DIFFERENT run is an unrelated update, not our hand-off', async () => {
    const home = tmpHome('ack-other-run')
    await claimantChild(markerPath(home), { startedAt: nowSeconds(), run: 'desk-1-zzz-ffff' })

    assert.deepEqual(await waitForHandoffClaim(home, process.pid, { runId: RUN, timeoutMs: 300, pollMs: 50 }), {
      taken: false
    })
  })

  test('a live claimant with our run but NO verifiable creation time is not acknowledged', async () => {
    const home = tmpHome('ack-no-ct')
    await claimantChild(markerPath(home), { startedAt: nowSeconds(), run: RUN, withCt: false })

    assert.deepEqual(await waitForHandoffClaim(home, process.pid, { runId: RUN, timeoutMs: 300, pollMs: 50 }), {
      taken: false
    })
  })

  test('a live claimant with our run but a reused-pid creation time is not acknowledged', async () => {
    const home = tmpHome('ack-reused')
    await claimantChild(markerPath(home), { startedAt: nowSeconds(), run: RUN, ctOffset: -3600 })

    assert.deepEqual(await waitForHandoffClaim(home, process.pid, { runId: RUN, timeoutMs: 300, pollMs: 50 }), {
      taken: false
    })
  })

  test('the Desktop own bridge carrying the run is never its own acknowledgment', async () => {
    const home = tmpHome('ack-self')
    assert.ok((await claimBridgeMarker(home, { startedAt: nowSeconds(), runId: RUN })).ok)

    assert.deepEqual(await waitForHandoffClaim(home, process.pid, { runId: RUN, timeoutMs: 300, pollMs: 50 }), {
      taken: false
    })
  })

  test('a live claimant with our run and its verified creation time IS the hand-off', async () => {
    const home = tmpHome('ack-taken')
    const startedAt = nowSeconds()
    await claimBridgeMarker(home, { startedAt, runId: RUN })
    const { child } = await claimantChild(markerPath(home), { startedAt, run: RUN })

    assert.deepEqual(await waitForHandoffClaim(home, process.pid, { runId: RUN, timeoutMs: 5_000, pollMs: 50 }), {
      taken: true,
      pid: child.pid
    })
  })

  test('the claimant is waited for while it starts, then acknowledged', async () => {
    const home = tmpHome('ack-late')
    const startedAt = nowSeconds()
    await claimBridgeMarker(home, { startedAt, runId: RUN })
    const waiting = waitForHandoffClaim(home, process.pid, { runId: RUN, timeoutMs: 10_000, pollMs: 50 })
    await new Promise(resolve => setTimeout(resolve, 200))
    const { child } = await claimantChild(markerPath(home), { startedAt, run: RUN })

    assert.deepEqual(await waiting, { taken: true, pid: child.pid })
  })
})

describe('legacy acknowledgment: new Desktop + script without protocol 2 (R4, SPEC 4b)', () => {
  test('a DEAD pid that echoed our started_at is not taken (the dead-PID hole stays closed)', async () => {
    const home = tmpHome('legacy-dead')
    const startedAt = nowSeconds()
    fs.writeFileSync(markerPath(home), `${await deadPid()}\n${startedAt}\n`)

    assert.deepEqual(await waitForHandoffClaim(home, process.pid, { startedAt, timeoutMs: 300, pollMs: 50 }), {
      taken: false
    })
  })

  test('a live pid that echoed our started_at IS taken (old scripts write `$$\\n$HERMES_UPDATE_STARTED_AT`)', async () => {
    const home = tmpHome('legacy-live')
    const startedAt = nowSeconds()
    const { child } = await claimantChild(markerPath(home), { startedAt, withCt: false })

    assert.deepEqual(await waitForHandoffClaim(home, process.pid, { startedAt, timeoutMs: 5_000, pollMs: 50 }), {
      taken: true,
      pid: child.pid
    })
  })

  test('a live pid carrying ANOTHER started_at is someone else, not our hand-off', async () => {
    const home = tmpHome('legacy-other')
    const startedAt = nowSeconds()
    await claimantChild(markerPath(home), { startedAt: startedAt - 30, withCt: false })

    assert.deepEqual(await waitForHandoffClaim(home, process.pid, { startedAt, timeoutMs: 300, pollMs: 50 }), {
      taken: false
    })
  })

  test.skipIf(!HAS_CT_PROBE)('a legacy v2 claim is ct-verified: a reused pid is not taken', async () => {
    const home = tmpHome('legacy-reused')
    const startedAt = nowSeconds()
    const { child } = await claimantChild(markerPath(home), { startedAt, ctOffset: -3600 })

    assert.ok(await processCreateTime(child.pid))
    assert.deepEqual(await waitForHandoffClaim(home, process.pid, { startedAt, timeoutMs: 300, pollMs: 50 }), {
      taken: false
    })
  })

  test('the Desktop own pid is never the successor', async () => {
    const home = tmpHome('legacy-self')
    const startedAt = nowSeconds()
    fs.writeFileSync(markerPath(home), `${process.pid}\n${startedAt}\n`)

    assert.deepEqual(await waitForHandoffClaim(home, process.pid, { startedAt, timeoutMs: 300, pollMs: 50 }), {
      taken: false
    })
  })
})
