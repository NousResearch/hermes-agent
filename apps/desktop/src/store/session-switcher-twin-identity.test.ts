/**
 * The session switcher is a THIRD entry point to the same session list the
 * Sessions sidebar shows, so it must agree with that surface about what a
 * session IS.
 *
 * Stored ids are only unique PER PROFILE: two profiles can hold sessions with
 * the same stored id (restored backups, copied state.dbs, cross-profile
 * imports — #92454), and the cross-profile list really does carry both as
 * distinct rows (`$sessions` in `store/session.ts`). The sidebar therefore
 * identifies a row by (profile, id) and pins that row's own
 * (connection, profile) as the resume owner before navigating
 * (`app/contrib/wiring.tsx::openStoredSession`) — otherwise the resume dials
 * the ambient backend and the transcript never loads (#82527).
 *
 * The switcher is the one surface that still reduced a row to its bare id, so
 * with twins present it could highlight/open the wrong profile's chat: the row
 * it picked was not the row the user was pointing at. These tests assert the
 * RELATION between the two surfaces — the switcher's target must resolve to
 * the same owner the sidebar pins for that same row — never a rendered
 * snapshot.
 *
 * Regression for #129518.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import {
  $selectedStoredSessionId,
  $sessions,
  _resetSessionOwnerHintsForTests,
  sessionOwnerRouteFromRow,
  setSessionOwnerHint
} from './session'
import {
  $switcherIndex,
  $switcherOpen,
  $switcherSessions,
  closeSwitcher,
  commitOnCtrlUp,
  onSwitcherTabDown,
  onSwitcherTabUp,
  openOrAdvanceSwitcher,
  slotSession,
  switcherRowKey
} from './session-switcher'

/** Two profiles, two connections, ONE shared stored id — the #92454 twin shape. */
const TWIN_ID = '20260101_shared'

const twinRow = (profile: string, connectionId: string, title: string): SessionInfo =>
  ({ connection_id: connectionId, id: TWIN_ID, profile, title }) as SessionInfo

const rowA = twinRow('alpha', 'source-a', 'Alpha planning')
const rowB = twinRow('beta', 'source-b', 'Beta planning')

const seed = (rows: SessionInfo[], selected: null | string) => {
  $sessions.set(rows)
  $selectedStoredSessionId.set(selected)
}

/** The owner the Sessions sidebar pins for a row — the contract the switcher
 *  has to match, read from the one shared resolver rather than restated. */
const sidebarOwnerFor = (row: SessionInfo) => sessionOwnerRouteFromRow(row)

beforeEach(() => {
  closeSwitcher()
  $switcherSessions.set([])
  $switcherIndex.set(0)
  seed([], null)
})

afterEach(() => {
  seed([], null)
  $switcherSessions.set([])
  _resetSessionOwnerHintsForTests()
})

describe('the session switcher agrees with the Sessions sidebar about row identity', () => {
  it('arms twins as two distinct rows, not one collapsed id', () => {
    // Three rows so the switcher's "fewer than two sessions" bail-out (a
    // quick-tap affordance, not an identity rule) cannot mask the assertion.
    seed([rowA, rowB, twinRow('alpha', 'source-a', 'Alpha later')], null)

    onSwitcherTabDown()
    openOrAdvanceSwitcher(1)
    onSwitcherTabUp()

    // Two rows share one stored id, so the list is longer than the set of ids.
    expect($switcherSessions.get().length).toBe(3)
    expect(new Set($switcherSessions.get().map(row => row.id)).size).toBe(1)
  })

  it('resolves a numeric slot to the row the sidebar would open, owner route included', () => {
    seed([rowA, rowB, twinRow('alpha', 'source-a', 'Alpha later')], null)

    onSwitcherTabDown()
    openOrAdvanceSwitcher(1)
    onSwitcherTabUp()

    const bySlot = slotSession(2)

    // Identity: the slot names the row at that index, so a twin is reachable
    // rather than collapsed into its twin.
    expect(bySlot).toBe($switcherSessions.get()[1])
    // Consistency: and that row's owner is the one the Sessions sidebar pins
    // for it — the two surfaces must not resolve the same row differently.
    expect(sidebarOwnerFor(bySlot!)).toEqual(sidebarOwnerFor($switcherSessions.get()[1]))
  })

  it('resolves a commit to the highlighted row, not the first row sharing its id', () => {
    seed([rowA, rowB, twinRow('alpha', 'source-a', 'Alpha later')], null)

    onSwitcherTabDown()
    openOrAdvanceSwitcher(1)
    onSwitcherTabUp()
    onSwitcherTabDown()
    openOrAdvanceSwitcher(1)
    onSwitcherTabUp()

    // Point at the SECOND twin, whose owner is a different connection.
    $switcherIndex.set(1)

    const picked = commitOnCtrlUp()

    expect(picked).toBe($switcherSessions.get()[1])
    // Pinning row A's owner while the user is looking at row B is the bug:
    // the resume would dial profile A's backend for profile B's chat.
    expect(sidebarOwnerFor(picked!)).toEqual(sidebarOwnerFor(rowB))
    expect(sidebarOwnerFor(picked!)).not.toEqual(sidebarOwnerFor(rowA))
  })

  it('still collapses to one row when every row shares a profile and id', () => {
    // The single-profile case must not regress: one owner, no ambiguity.
    seed([rowA, rowA], null)

    onSwitcherTabDown()
    openOrAdvanceSwitcher(1)
    onSwitcherTabUp()

    expect(slotSession(1)).toBe($switcherSessions.get()[0])
  })

  it('opens the HUD the same way with twins present', () => {
    seed([rowA, rowB, twinRow('alpha', 'source-a', 'Alpha later')], null)

    onSwitcherTabDown()
    openOrAdvanceSwitcher(1)
    onSwitcherTabUp()
    onSwitcherTabDown()
    openOrAdvanceSwitcher(1)
    onSwitcherTabUp()

    expect($switcherOpen.get()).toBe(true)
    expect($switcherSessions.get()[$switcherIndex.get()]).toBeDefined()
  })

  it('advances off the twin the user is on, not the first row sharing its id', () => {
    seed([rowA, rowB, twinRow('alpha', 'source-a', 'Alpha later')], TWIN_ID)
    // Opening a row records its owner route, and that route is the only thing
    // that says WHICH twin the selection names (#82527).
    setSessionOwnerHint(TWIN_ID, { connectionId: 'source-b', profile: 'beta', targetProfile: 'beta' })

    onSwitcherTabDown()
    openOrAdvanceSwitcher(1)
    onSwitcherTabUp()

    // A bare-id start anchored on row A (index 0) and landed back on row B, so
    // the shortcut did nothing. Anchored on B, it advances to row C.
    expect($switcherIndex.get()).toBe(2)
    expect($switcherSessions.get()[$switcherIndex.get()]?.title).toBe('Alpha later')
  })

  it('keys two twins apart, so the key is not only held up by a comment', () => {
    expect(switcherRowKey(rowA)).not.toBe(switcherRowKey(rowB))
    // The same (profile, id) IS the same row: the collapse this PR moves away
    // from must survive for genuine duplicates.
    expect(switcherRowKey(rowA)).toBe(switcherRowKey(twinRow('alpha', 'source-a', 'Alpha later')))
  })
})
