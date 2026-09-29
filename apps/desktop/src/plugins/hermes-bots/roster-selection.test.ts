/**
 * Selection reconciliation is the one place a RETIRED bot choice is dropped.
 * The defect these tests pin: dropping the dead key was not enough — the very
 * next render found no selection and seated the first surviving bot, silently
 * redirecting a retired bot-builder onto whatever happened to sort first. The
 * deferral must hold across renders (and reloads), and only an explicit user
 * choice may end it.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

const { hostMock } = vi.hoisted(() => ({
  hostMock: {
    request: vi.fn(),
    state: {
      connectionId: { get: vi.fn(() => 'local') },
      focusedSessionProfile: undefined,
      profile: { get: () => 'default' }
    }
  }
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return {
    atom,
    host: hostMock,
    queryClient: { getQueryData: vi.fn(), invalidateQueries: vi.fn(), setQueryData: vi.fn() },
    useQuery: vi.fn(),
    useValue: (store: { get: () => unknown }) => store.get()
  }
})
vi.mock('./shared', () => ({ getPluginCtx: () => null, ID: 'hermes-bots' }))

import {
  $rosterHydrated,
  $rosterSelectionDeferred,
  $selectedRosterHydrated,
  $selectedRosterKey,
  saveSelectedRosterBot
} from './bot-state'
import { reconcileRosterSelection, rosterWithSelectedOwner } from './roster-selection'

const ROSTER: RosterRow[] = [{ connectionId: 'local', name: 'ai-specialist' }]
const LOCAL_SOURCE = [{ connectionId: 'local', inventoryComplete: true, reachable: true }]

beforeEach(() => {
  $selectedRosterKey.set('')
  $rosterSelectionDeferred.set(false)
  $rosterHydrated.set(true)
  $selectedRosterHydrated.set(true)
})

describe('reconcileRosterSelection', () => {
  it('leaves the roster unselected AND stays that way on the next render after a retired choice', () => {
    $selectedRosterKey.set('local::bot-builder')

    reconcileRosterSelection(ROSTER, LOCAL_SOURCE, {}, Date.now())

    expect($selectedRosterKey.get()).toBe('')
    expect($rosterSelectionDeferred.get()).toBe(true)

    // Next render (and every one after a reload): the roster is still holding
    // the choice open instead of seating the first survivor.
    reconcileRosterSelection(ROSTER, LOCAL_SOURCE, {}, Date.now())

    expect($selectedRosterKey.get()).toBe('')
    expect($rosterSelectionDeferred.get()).toBe(true)
  })

  it('seats the first bot again once the user has chosen one explicitly', () => {
    $selectedRosterKey.set('local::bot-builder')

    reconcileRosterSelection(ROSTER, LOCAL_SOURCE, {}, Date.now())
    expect($rosterSelectionDeferred.get()).toBe(true)

    // The user's own pick ends the deferral...
    saveSelectedRosterBot(ROSTER[0])
    expect($rosterSelectionDeferred.get()).toBe(false)

    // ...and the roster may auto-seat again for a window that has nothing stored.
    $selectedRosterKey.set('')
    reconcileRosterSelection(ROSTER, LOCAL_SOURCE, {}, Date.now())

    expect($selectedRosterKey.get()).toBe('local::ai-specialist')
  })

  it('keeps a live selection and never defers it', () => {
    $selectedRosterKey.set('local::ai-specialist')

    reconcileRosterSelection(ROSTER, LOCAL_SOURCE, {}, Date.now())

    expect($selectedRosterKey.get()).toBe('local::ai-specialist')
    expect($rosterSelectionDeferred.get()).toBe(false)
  })

  it('keeps the selection (and defers nothing) while its source is merely unreachable', () => {
    $selectedRosterKey.set('homelab::research')

    reconcileRosterSelection(
      ROSTER,
      [{ connectionId: 'homelab', inventoryComplete: false, reachable: false }],
      {},
      Date.now()
    )

    expect($selectedRosterKey.get()).toBe('homelab::research')
    expect($rosterSelectionDeferred.get()).toBe(false)
  })

  it('keeps a selection whose source only REMEMBERED its list (R1)', () => {
    saveSelectedRosterBot({ connectionId: 'homelab', name: 'research' })
    expect($selectedRosterKey.get()).toBe('homelab::research')

    // An ssh source is never enumerated live — main always reports
    // connect-on-demand — so its list is a cache and `inventoryComplete` is
    // false forever, while `reachable` is TRUE because it has a list. One poll
    // behind a just-created bot used to deselect the user's own pick and raise
    // the deferral on top of it.
    reconcileRosterSelection(
      ROSTER,
      [{ connectionId: 'homelab', inventoryComplete: false, reachable: true }],
      {},
      Date.now()
    )

    expect($selectedRosterKey.get()).toBe('homelab::research')
    expect($rosterSelectionDeferred.get()).toBe(false)
  })

  it('paints a remembered-source selection as a ghost instead of dropping it (R1)', () => {
    const painted = rosterWithSelectedOwner(
      ROSTER,
      [{ connectionId: 'homelab', inventoryComplete: false, reachable: true }],
      'homelab::research'
    )

    expect(painted.map(row => row.name)).toEqual(['ai-specialist', 'research'])
    expect(painted[1]?.ghost).toBe(true)
  })

  it('ignores an answer issued before the selection was made (R2)', () => {
    const beforeThePick = Date.now() - 1_000

    saveSelectedRosterBot({ connectionId: 'local', name: 'bot-builder' })

    // A cached or out-of-order answer sent BEFORE the user picked never saw the
    // pick, so its silence about the bot proves nothing.
    reconcileRosterSelection(ROSTER, LOCAL_SOURCE, {}, beforeThePick)

    expect($selectedRosterKey.get()).toBe('local::bot-builder')
    expect($rosterSelectionDeferred.get()).toBe(false)

    // The next answer postdates the pick, so it IS a verdict the pick can be
    // held to: a genuinely retired bot is still dropped and deferred.
    reconcileRosterSelection(ROSTER, LOCAL_SOURCE, {}, Date.now())

    expect($selectedRosterKey.get()).toBe('')
    expect($rosterSelectionDeferred.get()).toBe(true)
  })

  it('never reconciles an undated answer (R2)', () => {
    saveSelectedRosterBot({ connectionId: 'local', name: 'bot-builder' })

    // Undated: it cannot be compared against the choice it would clear...
    reconcileRosterSelection(ROSTER, LOCAL_SOURCE, {}, undefined)

    expect($selectedRosterKey.get()).toBe('local::bot-builder')
    expect($rosterSelectionDeferred.get()).toBe(false)

    // ...and it must not seat one either.
    saveSelectedRosterBot(ROSTER[0])
    $selectedRosterKey.set('')

    reconcileRosterSelection(ROSTER, LOCAL_SOURCE, {}, undefined)

    expect($selectedRosterKey.get()).toBe('')
  })
})
