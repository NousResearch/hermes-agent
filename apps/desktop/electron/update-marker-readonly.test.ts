/**
 * A7 rule 3: Electron NEVER deletes or rewrites the update marker. Reading
 * and judging a dead marker — at boot, before a hand-off, before a bridge or
 * staged-updater pre-write — leaves its bytes exactly as they were; reclaiming
 * belongs to the checkout script helper under the sidecar lock. (The previous
 * head compare-and-deleted a dead marker from all of these paths.)
 */

import fs from 'fs'
import assert from 'node:assert/strict'

import { afterEach, test } from 'vitest'

import {
  claimBridgeMarker,
  EMPTY_MARKER_GRACE_MS,
  markerPath,
  readLiveUpdateMarker,
  updateHandoffConflict,
  writeUpdateMarker
} from './update-marker'
import { cleanupMarkerFixtures, deadPid, minutesAgo, tmpHome } from './update-marker.test-helpers'

afterEach(cleanupMarkerFixtures)

async function deadMarkerBodies(): Promise<[string, string][]> {
  const gone = await deadPid()

  return [
    ['dead v2 owner', `${gone}\n${minutesAgo(1)}\nct:1700000000.000\n`],
    ['dead v1 owner', `${gone}\n${minutesAgo(1)}\n`],
    [
      'dead owner + dead delegate (crash-12db shape)',
      `${gone}\n${minutesAgo(1)}\nct:1.000\ndelegate:${gone} ct:2.000\n`
    ],
    ['malformed', 'not-a-pid\nnonsense\n'],
    ['previous incarnation of our own pid', `${process.pid}\n${minutesAgo(1)}\n`]
  ]
}

test('readLiveUpdateMarker on a dead marker reports "not running" and leaves the bytes', async () => {
  for (const [label, body] of await deadMarkerBodies()) {
    const home = tmpHome('ro-read')
    fs.writeFileSync(markerPath(home), body)

    assert.equal(await readLiveUpdateMarker(home), null, label)
    assert.equal(fs.readFileSync(markerPath(home), 'utf8'), body, `${label}: bytes unchanged`)
  }
})

test('updateHandoffConflict on a dead marker reports no conflict and leaves the bytes', async () => {
  for (const [label, body] of await deadMarkerBodies()) {
    const home = tmpHome('ro-conflict')
    fs.writeFileSync(markerPath(home), body)

    assert.equal(await updateHandoffConflict(home), null, label)
    assert.equal(fs.readFileSync(markerPath(home), 'utf8'), body, `${label}: bytes unchanged`)
  }
})

test('claimBridgeMarker over a dead marker does not reclaim it: no claim, bytes unchanged', async () => {
  for (const [label, body] of await deadMarkerBodies()) {
    const home = tmpHome('ro-bridge')
    fs.writeFileSync(markerPath(home), body)

    const claim = await claimBridgeMarker(home, { startedAt: 1 })

    assert.equal(claim.ok, false, `${label}: the exclusive create loses to any existing marker`)
    assert.equal(fs.readFileSync(markerPath(home), 'utf8'), body, `${label}: bytes unchanged`)
  }
})

test('writeUpdateMarker (staged pre-write) over a dead marker skips: bytes unchanged', async () => {
  for (const [label, body] of await deadMarkerBodies()) {
    const home = tmpHome('ro-prewrite')
    fs.writeFileSync(markerPath(home), body)

    const prewrite = await writeUpdateMarker(home, process.pid, { startedAt: 1 })

    assert.equal(prewrite.ok, false, label)
    assert.equal(fs.readFileSync(markerPath(home), 'utf8'), body, `${label}: bytes unchanged`)
  }
})

test('a stale 0-byte marker is judged dead and left in place', async () => {
  const home = tmpHome('ro-empty')
  const file = markerPath(home)
  fs.writeFileSync(file, '')
  const old = (Date.now() - EMPTY_MARKER_GRACE_MS - 5000) / 1000
  fs.utimesSync(file, old, old)

  assert.equal(await readLiveUpdateMarker(home), null)
  assert.equal(await updateHandoffConflict(home), null)
  assert.equal((await claimBridgeMarker(home, { startedAt: 1 })).ok, false)
  assert.ok(fs.existsSync(file))
  assert.equal(fs.statSync(file).size, 0)
})
