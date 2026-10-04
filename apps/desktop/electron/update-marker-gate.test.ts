/**
 * The update gate's marker probe (R6): a dead/malformed marker is routed
 * through the checkout script's `reclaim` helper — never deleted by Electron —
 * and `held` (a completion still holds the checkout lock) keeps the boot
 * parked. The helper runs as a REAL fake script process.
 */

import fs from 'fs'
import assert from 'node:assert/strict'
import path from 'path'

import { afterEach, describe, test } from 'vitest'

import { markerPath } from './update-marker'
import { liveMarkerProbe } from './update-marker-gate'
import { cleanupMarkerFixtures, deadPid, liveOwner, minutesAgo, tmpHome } from './update-marker.test-helpers'
import { runMarkerHelper } from './updater/marker-helper'
import { cleanupFakeCheckouts, fakeHelperCheckout } from './updater/marker-helper.test-helpers'

afterEach(() => {
  cleanupMarkerFixtures()
  cleanupFakeCheckouts()
})

function helperCalls(home: string): string[] {
  try {
    return fs.readFileSync(path.join(home, 'helper-calls.log'), 'utf8').trim().split('\n').filter(Boolean)
  } catch {
    return []
  }
}

describe.skipIf(process.platform === 'win32')('gate over a dead marker (R6)', () => {
  function gate(root: string, home: string, now?: () => number) {
    return liveMarkerProbe({
      hermesHome: home,
      reclaim: () => runMarkerHelper('reclaim', { updateRoot: root, hermesHome: home, isWindows: false }),
      now
    })
  }

  test('`held` keeps the gate closed and is re-asked every 5 s; the marker is never touched', async () => {
    const { root, home } = fakeHelperCheckout()
    const body = `${await deadPid()}\n${minutesAgo(1)}\nct:1.000\n`
    fs.writeFileSync(markerPath(home), body)
    fs.writeFileSync(path.join(home, 'helper-verdict'), 'held')
    let clock = Date.now()
    const hasLiveMarker = gate(root, home, () => clock)

    assert.equal(await hasLiveMarker(), true, 'a completion still holds the checkout: keep waiting')
    assert.equal(await hasLiveMarker(), true)
    assert.equal(helperCalls(home).length, 1, 'not one helper spawn per poll')

    clock += 5_000
    fs.writeFileSync(path.join(home, 'helper-verdict'), 'busy')
    assert.equal(await hasLiveMarker(), true)
    assert.equal(helperCalls(home).length, 2, 'held is re-probed after 5 s')
    assert.equal(fs.readFileSync(markerPath(home), 'utf8'), body)

    clock += 5_000
    fs.writeFileSync(path.join(home, 'helper-verdict'), 'reclaimed')
    assert.equal(await hasLiveMarker(), false, 'once the script reclaims, the gate opens')
    assert.equal(fs.existsSync(markerPath(home)), false, 'the SCRIPT removed it, under its lock')
  })

  test('`live <pid>` from the helper keeps the gate closed', async () => {
    const { root, home } = fakeHelperCheckout()
    fs.writeFileSync(markerPath(home), 'garbage\n')
    fs.writeFileSync(path.join(home, 'helper-verdict'), 'live 4242')

    assert.equal(await gate(root, home)(), true)
  })

  test('`unsupported` (old checkout) opens the gate WITHOUT deleting; asked once per distinct body', async () => {
    const { root, home } = fakeHelperCheckout()
    const body = `${await deadPid()}\n${minutesAgo(1)}\n`
    fs.writeFileSync(markerPath(home), body)
    fs.writeFileSync(path.join(home, 'helper-verdict'), 'usage: unknown option --marker-op')
    fs.writeFileSync(path.join(home, 'helper-exit'), '64')
    const hasLiveMarker = gate(root, home)

    assert.equal(await hasLiveMarker(), false)
    assert.equal(await hasLiveMarker(), false)
    assert.equal(helperCalls(home).length, 1)
    assert.equal(fs.readFileSync(markerPath(home), 'utf8'), body, 'dead = not running, and left in place')

    const next = `${await deadPid()}\n${minutesAgo(1)}\n`
    fs.writeFileSync(markerPath(home), next)
    assert.equal(await hasLiveMarker(), false)
    assert.equal(helperCalls(home).length, 2, 'a different dead body is asked about again')
  })

  test('a live owner never reaches the helper; an absent marker neither', async () => {
    const { root, home } = fakeHelperCheckout()
    const hasLiveMarker = gate(root, home)

    assert.equal(await hasLiveMarker(), false)
    const owner = await liveOwner()
    fs.writeFileSync(markerPath(home), `${owner.pid}\n${minutesAgo(1)}\n`)
    assert.equal(await hasLiveMarker(), true)
    assert.deepEqual(helperCalls(home), [])
  })
})

test('without a protocol-2 script the gate judges dead = not running and deletes nothing', async () => {
  const home = tmpHome('gate-legacy')
  const body = `${await deadPid()}\n${minutesAgo(1)}\n`
  fs.writeFileSync(markerPath(home), body)

  assert.equal(await liveMarkerProbe({ hermesHome: home, reclaim: null })(), false)
  assert.equal(fs.readFileSync(markerPath(home), 'utf8'), body)
})
