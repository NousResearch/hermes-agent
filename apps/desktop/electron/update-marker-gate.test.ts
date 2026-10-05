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
import {
  allowStartOverHold,
  type HeldState,
  heldWaitMessage,
  HOLD_SCREEN_GRACE_MS,
  holdTicker,
  liveMarkerProbe,
  PRIMARY_HOLD_OWNER,
  requestHoldRecheck,
  resetHoldStateForTests,
  UpdateHoldBoard
} from './update-marker-gate'
import { cleanupMarkerFixtures, deadPid, liveOwner, minutesAgo, tmpHome } from './update-marker.test-helpers'
import { runMarkerHelper } from './updater/marker-helper'
import { cleanupFakeCheckouts, fakeHelperCheckout } from './updater/marker-helper.test-helpers'

// The old held ceiling (scripts' RELEASE_WAIT_S); the gate must not honour it.
const OLD_HELD_CEILING_MS = 7_200_000

afterEach(() => {
  resetHoldStateForTests()
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
  function gate(
    root: string,
    home: string,
    now?: () => number,
    extra: {
      onHeld?: (state: HeldState) => void
      log?: (line: string) => void
      onOverride?: (holdId: string) => void
    } = {}
  ) {
    return liveMarkerProbe({
      hermesHome: home,
      reclaim: () => runMarkerHelper('reclaim', { updateRoot: root, hermesHome: home, isWindows: false }),
      now,
      ...extra
    })
  }

  test('`held` never opens the gate by itself, long past the old 7200 s ceiling; Retry re-asks at once (R8 D3)', async () => {
    const { root, home } = fakeHelperCheckout()
    const owner = await deadPid()
    const body = `${owner}\n${minutesAgo(1)}\nct:1.000\n`
    fs.writeFileSync(markerPath(home), body)
    fs.writeFileSync(path.join(home, 'helper-verdict'), 'held')
    let clock = Date.now()
    const states: HeldState[] = []
    const hasLiveMarker = gate(root, home, () => clock, { onHeld: s => states.push(s) })

    assert.equal(await hasLiveMarker(), true, 'held: parked')
    clock += 10 * OLD_HELD_CEILING_MS
    assert.equal(await hasLiveMarker(), true, 'still parked: a hold is never aged out')
    assert.equal(await gate(root, home, () => clock)(), true, 'a later gate wait is parked too')
    assert.equal(fs.readFileSync(markerPath(home), 'utf8'), body, 'Desktop never mutates the marker (A7)')

    const [first] = states
    assert.deepEqual([first.verdict, first.ownerPid, first.blocking], ['held', owner, true])
    assert.equal(states.at(-1)!.holdId, first.holdId, 'one hold id per marker body')
    assert.match(heldWaitMessage(first), new RegExp(`update \\(process ${owner}\\) exited.*still holds`))

    const calls = helperCalls(home).length
    assert.equal(await hasLiveMarker(), true)
    assert.equal(helperCalls(home).length, calls, 'inside the re-probe interval: not re-asked')
    requestHoldRecheck()
    fs.writeFileSync(path.join(home, 'helper-verdict'), 'reclaimed')
    assert.equal(await hasLiveMarker(), false, 'Retry re-asks at once; the hold ended, so the gate opens')
    assert.equal(helperCalls(home).length, calls + 1)
  })

  test('a confirmed Start anyway opens only the hold it names, for every wait; a new marker body blocks again (R8 D3)', async () => {
    const { root, home } = fakeHelperCheckout()
    const body = `${await deadPid()}\n${minutesAgo(1)}\nct:1.000\n`
    fs.writeFileSync(markerPath(home), body)
    fs.writeFileSync(path.join(home, 'helper-verdict'), 'held')
    const states: HeldState[] = []
    const logs: string[] = []
    const overrides: string[] = []

    const primary = gate(root, home, undefined, {
      onHeld: s => states.push(s),
      log: l => logs.push(l),
      onOverride: id => overrides.push(id)
    })

    assert.equal(await primary(), true)
    allowStartOverHold(states[0].holdId)
    assert.equal(await primary(), false, 'the confirmed hold no longer blocks this wait')
    assert.equal(await primary(), false)
    assert.deepEqual(
      overrides,
      [states[0].holdId, states[0].holdId],
      'the waiter learns it ended by override, not finish'
    )
    assert.equal(await gate(root, home)(), false, 'nor a pool backend wait in the same process')
    assert.equal(logs.filter(l => l.includes('chose Start anyway')).length, 1, 'the override is logged once per wait')
    assert.equal(fs.readFileSync(markerPath(home), 'utf8'), body, 'the marker stays in place')

    fs.writeFileSync(markerPath(home), `${await deadPid()}\n${minutesAgo(1)}\nct:2.000\n`)
    assert.equal(await primary(), true, 'a different marker body is a different hold: blocked again')
  })

  test('a held body never opens on `unsupported`: a script missing mid-update blocks and is re-asked (R8 M5)', async () => {
    const { root, home } = fakeHelperCheckout()
    fs.writeFileSync(markerPath(home), `${await deadPid()}\n${minutesAgo(1)}\nct:1.000\n`)
    fs.writeFileSync(path.join(home, 'helper-verdict'), 'held')
    const script = path.join(root, 'scripts', 'desktop-update', 'posix.sh')
    const text = fs.readFileSync(script, 'utf8')
    const states: HeldState[] = []
    const probe = gate(root, home, undefined, { onHeld: s => states.push(s) })

    assert.equal(await probe(), true)
    fs.rmSync(script) // git unlinks a changed script before writing the new one
    requestHoldRecheck()
    assert.equal(await probe(), true, 'the body that was held stays blocked')
    assert.deepEqual([states.at(-1)!.verdict, states.at(-1)!.blocking], ['error', true])
    fs.writeFileSync(script, text)
    requestHoldRecheck()
    assert.equal(await probe(), true)
    assert.equal(states.at(-1)!.verdict, 'held', 're-asked once the script is back')
    fs.writeFileSync(path.join(home, 'helper-verdict'), 'reclaimed')
    requestHoldRecheck()
    assert.equal(await probe(), false, 'only the script reclaiming opens it')
  })

  test('two empty dead markers are two holds: Start anyway over one does not pass the next (R8 m7)', async () => {
    const { root, home } = fakeHelperCheckout()
    fs.writeFileSync(path.join(home, 'helper-verdict'), 'held')

    const emptyMarker = (agoS: number) => {
      fs.rmSync(markerPath(home), { force: true })
      fs.writeFileSync(markerPath(home), '')
      const at = Date.now() / 1000 - agoS
      fs.utimesSync(markerPath(home), at, at)
    }

    const states: HeldState[] = []
    emptyMarker(600)
    assert.equal(await gate(root, home, undefined, { onHeld: s => states.push(s) })(), true)
    allowStartOverHold(states[0].holdId)
    assert.equal(await gate(root, home)(), false, 'the confirmed empty marker passes')
    emptyMarker(300) // a later update's empty claim, also past the grace
    assert.equal(await gate(root, home)(), true, 'a different marker file is a different hold')
  })

  test('a failing protocol-2 helper keeps the gate parked and is retried after recovery', async () => {
    const { root, home } = fakeHelperCheckout()
    const body = `${await deadPid()}\n${minutesAgo(1)}\nct:1.000\n`
    fs.writeFileSync(markerPath(home), body)
    fs.writeFileSync(path.join(home, 'helper-verdict'), 'held')
    fs.writeFileSync(path.join(home, 'helper-exit'), '1')
    let clock = Date.now()
    const probe = gate(root, home, () => clock)
    assert.equal(await probe(), true, 'an operational failure is not legacy capability absence')
    fs.writeFileSync(path.join(home, 'helper-exit'), '0')
    clock += 5_000
    assert.equal(await probe(), true)
    assert.equal(helperCalls(home).length, 2, 'same waiter retries the failed helper')
    fs.writeFileSync(path.join(home, 'helper-verdict'), 'reclaimed')
    clock += 5_000
    assert.equal(await probe(), false)
  })

  test('sidecar contention over an old marker is indeterminate, not an expired held checkout', async () => {
    const { root, home } = fakeHelperCheckout()
    fs.writeFileSync(markerPath(home), `${await deadPid()}\n${minutesAgo(180)}\nct:1.000\n`)
    fs.writeFileSync(path.join(home, 'helper-verdict'), 'busy')
    const states: HeldState[] = []
    assert.equal(await gate(root, home, undefined, { onHeld: s => states.push(s) })(), true)
    assert.equal(states[0].blocking, true, 'blocks like held: the boot screen offers the same ways out')
    assert.match(heldWaitMessage(states[0]), /verify.*ownership/)
  })

  test('a helper `live <pid>` names a live identity: waited out, not a blocked screen', async () => {
    const { root, home } = fakeHelperCheckout()
    fs.writeFileSync(markerPath(home), `${await deadPid()}\n${minutesAgo(1)}\nct:1.000\n`)
    fs.writeFileSync(path.join(home, 'helper-verdict'), 'live 4242')
    let clock = Date.now()
    const states: HeldState[] = []
    const hasLiveMarker = gate(root, home, () => clock, { onHeld: s => states.push(s) })

    assert.equal(await hasLiveMarker(), true)
    clock += 10 * OLD_HELD_CEILING_MS
    assert.equal(await hasLiveMarker(), true)
    assert.equal(states.at(-1)!.blocking, false)
    assert.equal(states.at(-1)!.livePid, 4242)
  })

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
    const { root, home } = fakeHelperCheckout('')
    const body = `${await deadPid()}\n${minutesAgo(1)}\n`
    fs.writeFileSync(markerPath(home), body)
    fs.writeFileSync(path.join(home, 'helper-verdict'), 'usage: unknown option --marker-op')
    fs.writeFileSync(path.join(home, 'helper-exit'), '64')
    const hasLiveMarker = gate(root, home)

    assert.equal(await hasLiveMarker(), false)
    assert.equal(await hasLiveMarker(), false)
    assert.equal(helperCalls(home).length, 0, 'legacy helper is never invoked')
    assert.equal(fs.readFileSync(markerPath(home), 'utf8'), body, 'dead = not running, and left in place')

    const next = `${await deadPid()}\n${minutesAgo(1)}\n`
    fs.writeFileSync(markerPath(home), next)
    assert.equal(await hasLiveMarker(), false)
    assert.equal(helperCalls(home).length, 0, 'legacy capability absence remains unsupported')
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

test("a pool/profile hold reaches the screen when the primary shows none; the primary's own clear never drops it (R8 M6)", () => {
  const state = (holdId: string): HeldState => ({
    verdict: 'held',
    ownerPid: 7,
    livePid: null,
    holdId,
    since: 1,
    checkedAt: 2,
    blocking: true
  })

  const board = new UpdateHoldBoard()
  assert.equal(board.shown(), null)
  board.set('pool:work', state('p'))
  assert.equal(board.shown()?.holdId, 'p', 'a remote primary / an already-booted primary: the pool hold is shown')
  board.set(PRIMARY_HOLD_OWNER, state('m'))
  assert.equal(board.shown()?.holdId, 'm', 'the primary boot wait wins')
  board.clear(PRIMARY_HOLD_OWNER)
  assert.equal(board.shown()?.holdId, 'p', 'clearing the primary leaves the pool hold on screen')
  board.clear('pool:work')
  assert.equal(board.shown(), null)
})

test('without a protocol-2 script the gate judges dead = not running and deletes nothing', async () => {
  const home = tmpHome('gate-legacy')
  const body = `${await deadPid()}\n${minutesAgo(1)}\n`
  fs.writeFileSync(markerPath(home), body)

  assert.equal(await liveMarkerProbe({ hermesHome: home, reclaim: null })(), false)
  assert.equal(fs.readFileSync(markerPath(home), 'utf8'), body)
})

// Review 5411222842 (shape): the boot and pool waits share one hold state
// machine. A blocking hold shows only after an unbroken grace, any break
// restarts the grace and clears this wait's hold, and nothing ever times out.
describe('holdTicker', () => {
  const state = (blocking: boolean): HeldState => ({
    verdict: 'held',
    ownerPid: 7,
    livePid: null,
    holdId: 'h1',
    since: 0,
    checkedAt: 0,
    blocking
  })

  test('shows a blocking hold after an unbroken grace and clears on any break', () => {
    let clock = 0
    const shown: string[] = []

    const hold = holdTicker({
      show: s => shown.push(`show:${s.holdId}`),
      clear: () => shown.push('clear'),
      now: () => clock
    })

    hold.onHeld(state(true))
    assert.deepEqual(hold.tick('marker'), { held: state(true), shown: false })
    clock += HOLD_SCREEN_GRACE_MS
    hold.onHeld(state(true))
    assert.equal(hold.tick('marker').shown, true)
    // A poll with no held verdict (the probe saw a live owner) breaks the hold.
    assert.deepEqual(hold.tick('marker'), { held: null, shown: false })
    hold.onHeld(state(true))
    clock += HOLD_SCREEN_GRACE_MS - 1
    assert.equal(hold.tick('marker').shown, false, 'the grace restarts after a break')
    clock += 10 * OLD_HELD_CEILING_MS
    hold.onHeld(state(true))
    assert.equal(hold.tick('marker').shown, true, 'never a timeout: still parked behind the screen')
    assert.deepEqual(shown, ['clear', 'show:h1', 'clear', 'clear', 'show:h1'])
  })

  test('a non-blocking hold or a non-marker reason never shows the screen', () => {
    let clock = 0
    const hold = holdTicker({ show: () => assert.fail('shown'), clear: () => {}, now: () => clock })

    for (const [reason, blocking] of [
      ['marker', false],
      ['in-flight', true]
    ] as const) {
      hold.onHeld(state(blocking))
      hold.tick(reason)
      clock += HOLD_SCREEN_GRACE_MS * 2
      hold.onHeld(state(blocking))
      assert.equal(hold.tick(reason).shown, false)
    }
  })
})
