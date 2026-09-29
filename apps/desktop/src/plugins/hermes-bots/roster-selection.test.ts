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
import { reconcileRosterSelection } from './roster-selection'

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

    reconcileRosterSelection(ROSTER, LOCAL_SOURCE, {})

    expect($selectedRosterKey.get()).toBe('')
    expect($rosterSelectionDeferred.get()).toBe(true)

    // Next render (and every one after a reload): the roster is still holding
    // the choice open instead of seating the first survivor.
    reconcileRosterSelection(ROSTER, LOCAL_SOURCE, {})

    expect($selectedRosterKey.get()).toBe('')
    expect($rosterSelectionDeferred.get()).toBe(true)
  })

  it('seats the first bot again once the user has chosen one explicitly', () => {
    $selectedRosterKey.set('local::bot-builder')

    reconcileRosterSelection(ROSTER, LOCAL_SOURCE, {})
    expect($rosterSelectionDeferred.get()).toBe(true)

    // The user's own pick ends the deferral...
    saveSelectedRosterBot(ROSTER[0])
    expect($rosterSelectionDeferred.get()).toBe(false)

    // ...and the roster may auto-seat again for a window that has nothing stored.
    $selectedRosterKey.set('')
    reconcileRosterSelection(ROSTER, LOCAL_SOURCE, {})

    expect($selectedRosterKey.get()).toBe('local::ai-specialist')
  })

  it('keeps a live selection and never defers it', () => {
    $selectedRosterKey.set('local::ai-specialist')

    reconcileRosterSelection(ROSTER, LOCAL_SOURCE, {})

    expect($selectedRosterKey.get()).toBe('local::ai-specialist')
    expect($rosterSelectionDeferred.get()).toBe(false)
  })

  it('keeps the selection (and defers nothing) while its source is merely unreachable', () => {
    $selectedRosterKey.set('homelab::research')

    reconcileRosterSelection(ROSTER, [{ connectionId: 'homelab', inventoryComplete: false, reachable: false }], {})

    expect($selectedRosterKey.get()).toBe('homelab::research')
    expect($rosterSelectionDeferred.get()).toBe(false)
  })
})
