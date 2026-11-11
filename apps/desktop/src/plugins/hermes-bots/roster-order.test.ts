import type { PluginContext } from '@hermes/plugin-sdk'
import { beforeEach, describe, expect, it } from 'vitest'

import {
  $rosterOrder,
  clearRosterOrder,
  clearSectionRosterOrder,
  hasAnyCustomRosterOrder,
  loadRosterOrder,
  moveRosterItem,
  normalizeRosterOrderMap,
  orderRosterRows,
  orderSectionMixedRows,
  persistRosterOrder,
  pruneRosterOrder,
  removeBotFromRosterOrder,
  removeBotFromSectionOrder,
  restoreSectionRosterOrder,
  rosterSectionScope,
  snapshotSectionRosterOrder
} from './roster-order'
import { setPluginCtx } from './shared'
import type { GatewaySource, RosterRow } from './types'

describe('roster-order (section-scoped manual ordering)', () => {
  let mockStorage: Record<string, unknown> = {}

  beforeEach(() => {
    mockStorage = {}
    setPluginCtx({
      storage: {
        get: (key: string, fallback: unknown) => (key in mockStorage ? mockStorage[key] : fallback),
        set: (key: string, val: unknown) => {
          mockStorage[key] = val
        }
      }
    } as unknown as PluginContext)
    $rosterOrder.set({})
  })

  describe('normalizeRosterOrderMap', () => {
    it('handles non-object inputs gracefully', () => {
      expect(normalizeRosterOrderMap(null)).toEqual({})
      expect(normalizeRosterOrderMap(undefined)).toEqual({})
      expect(normalizeRosterOrderMap('invalid')).toEqual({})
      expect(normalizeRosterOrderMap(123)).toEqual({})
      expect(normalizeRosterOrderMap(['array', 'not', 'map'])).toEqual({})
    })

    it('deduplicates, trims scope and bot keys, and drops empty lists', () => {
      const input = {
        ' local::sec-a ': ['bot-1', ' bot-2 ', 'bot-1', '', '  ', 'bot-3'],
        '   ': ['bot-ignored'],
        'local::empty': ['', '   '],
        'local::invalid': 'not-an-array'
      }

      expect(normalizeRosterOrderMap(input)).toEqual({
        'local::sec-a': ['bot-1', 'bot-2', 'bot-3']
      })
    })
  })

  describe('rosterSectionScope', () => {
    it('formats connection and section identity', () => {
      expect(rosterSectionScope('sec-123', 'remote-gw')).toBe('remote-gw::sec-123')
    })

    it('defaults connection to local when absent', () => {
      expect(rosterSectionScope('sec-123', null)).toBe('local::sec-123')
      expect(rosterSectionScope('sec-123', '')).toBe('local::sec-123')
    })

    it('defaults section to unassigned when null or empty', () => {
      expect(rosterSectionScope(null, 'remote-gw')).toBe('remote-gw::unassigned')
      expect(rosterSectionScope('', 'local')).toBe('local::unassigned')
      expect(rosterSectionScope(null, null)).toBe('local::unassigned')
    })
  })

  describe('orderRosterRows', () => {
    const mockRows = [
      { key: 'bot-1', activity: 100, created: 1000, pinned: false },
      { key: 'bot-2', activity: 200, created: 2000, pinned: false },
      { key: 'bot-3', activity: 50, created: 3000, pinned: false },
      { key: 'bot-pinned', activity: 10, created: 500, pinned: true }
    ]

    it('defaults to pinned first, then activity descending when orderKeys is empty', () => {
      const sorted = orderRosterRows(mockRows, r => r.key, [])
      expect(sorted.map(r => r.key)).toEqual(['bot-pinned', 'bot-2', 'bot-1', 'bot-3'])
    })

    it('respects custom manual order while keeping pinned items at top', () => {
      const sorted = orderRosterRows(mockRows, r => r.key, ['bot-3', 'bot-1'])
      expect(sorted.map(r => r.key)).toEqual(['bot-pinned', 'bot-3', 'bot-1', 'bot-2'])
    })

    it('orders multiple pinned items according to custom order', () => {
      const rowsWithTwoPinned = [
        { key: 'pin-a', activity: 10, created: 100, pinned: true },
        { key: 'pin-b', activity: 50, created: 200, pinned: true },
        { key: 'normal-1', activity: 100, created: 300, pinned: false }
      ]

      const sortedAB = orderRosterRows(rowsWithTwoPinned, r => r.key, ['pin-a', 'pin-b'])
      expect(sortedAB.map(r => r.key)).toEqual(['pin-a', 'pin-b', 'normal-1'])

      const sortedBA = orderRosterRows(rowsWithTwoPinned, r => r.key, ['pin-b', 'pin-a'])
      expect(sortedBA.map(r => r.key)).toEqual(['pin-b', 'pin-a', 'normal-1'])
    })

    it('places unranked newly discovered bots deterministically without activity churn', () => {
      const rowsWithNewBots = [
        { key: 'ranked-1', activity: 100, created: 1000, pinned: false },
        { key: 'ranked-2', activity: 200, created: 2000, pinned: false },
        { key: 'unranked-older', activity: 50, created: 3000, pinned: false },
        { key: 'unranked-newer', activity: 10, created: 4000, pinned: false }
      ]

      const sorted1 = orderRosterRows(rowsWithNewBots, r => r.key, ['ranked-1', 'ranked-2'])
      // Ranked items follow custom order; unranked follow created timestamp desc (newer first)
      expect(sorted1.map(r => r.key)).toEqual(['ranked-1', 'ranked-2', 'unranked-newer', 'unranked-older'])

      // Activity spike on unranked-older does NOT reorder unranked bots (prevents chat activity churn)
      const rowsAfterChatSpike = [
        { key: 'ranked-1', activity: 100, created: 1000, pinned: false },
        { key: 'ranked-2', activity: 200, created: 2000, pinned: false },
        { key: 'unranked-older', activity: 99999, created: 3000, pinned: false },
        { key: 'unranked-newer', activity: 10, created: 4000, pinned: false }
      ]

      const sorted2 = orderRosterRows(rowsAfterChatSpike, r => r.key, ['ranked-1', 'ranked-2'])
      expect(sorted2.map(r => r.key)).toEqual(['ranked-1', 'ranked-2', 'unranked-newer', 'unranked-older'])
    })
  })

  describe('moveRosterItem', () => {
    const allBots = ['a', 'b', 'c', 'd']

    it('moves item before target', () => {
      const next = moveRosterItem(['a', 'b', 'c', 'd'], allBots, 'd', 'b', 'before')
      expect(next).toEqual(['a', 'd', 'b', 'c'])
    })

    it('moves item after target', () => {
      const next = moveRosterItem(['a', 'b', 'c', 'd'], allBots, 'a', 'c', 'after')
      expect(next).toEqual(['b', 'c', 'a', 'd'])
    })

    it('returns unchanged order if fromKey equals toKey', () => {
      const next = moveRosterItem(['a', 'b', 'c'], allBots, 'b', 'b', 'before')
      expect(next).toEqual(['a', 'b', 'c'])
    })

    it('seeds full section bots on first drag when current order is empty', () => {
      const next = moveRosterItem([], ['x', 'y', 'z'], 'z', 'x', 'before')
      expect(next).toEqual(['z', 'x', 'y'])
    })

    it('preserves filtered/hidden bots in durable order during search projection', () => {
      const currentOrder = ['a', 'b', 'c', 'd']
      // Search matches only 'a' and 'd'
      // User moves 'd' before 'a'
      const next = moveRosterItem(currentOrder, allBots, 'd', 'a', 'before')
      // 'b' and 'c' are retained and not lost
      expect(next).toEqual(['d', 'a', 'b', 'c'])
    })
  })

  describe('interaction with #101290 sections', () => {
    it('scopes ordering to section: reordering Section A leaves Section B and Unassigned unchanged', () => {
      const scopeA = rosterSectionScope('sec-A', 'local')
      const scopeB = rosterSectionScope('sec-B', 'local')
      const scopeUnassigned = rosterSectionScope(null, 'local')

      const initialMap = {
        [scopeA]: ['bot-a1', 'bot-a2', 'bot-a3'],
        [scopeB]: ['bot-b1', 'bot-b2'],
        [scopeUnassigned]: ['bot-u1', 'bot-u2']
      }

      persistRosterOrder(initialMap)

      // Reorder inside Section A
      const nextOrderA = moveRosterItem(
        initialMap[scopeA],
        ['bot-a1', 'bot-a2', 'bot-a3'],
        'bot-a3',
        'bot-a1',
        'before'
      )

      persistRosterOrder({
        ...$rosterOrder.get(),
        [scopeA]: nextOrderA
      })

      const state = $rosterOrder.get()
      expect(state[scopeA]).toEqual(['bot-a3', 'bot-a1', 'bot-a2'])
      // Section B and Unassigned are completely untouched
      expect(state[scopeB]).toEqual(['bot-b1', 'bot-b2'])
      expect(state[scopeUnassigned]).toEqual(['bot-u1', 'bot-u2'])
    })

    it('supports reordering inside Unassigned section', () => {
      const scopeUnassigned = rosterSectionScope(null, 'local')
      const unassignedBots = ['u1', 'u2', 'u3']

      const next = moveRosterItem([], unassignedBots, 'u3', 'u1', 'before')
      persistRosterOrder({
        [scopeUnassigned]: next
      })

      expect($rosterOrder.get()[scopeUnassigned]).toEqual(['u3', 'u1', 'u2'])
    })

    it('prunes bot from old section order on move so it does not leak into new section', () => {
      const scopeA = rosterSectionScope('sec-A', 'local')
      const scopeB = rosterSectionScope('sec-B', 'local')

      persistRosterOrder({
        [scopeA]: ['bot-1', 'bot-2', 'bot-3'],
        [scopeB]: ['bot-4', 'bot-5']
      })

      // Move bot-1 out of Section A
      removeBotFromRosterOrder('bot-1', scopeA)

      const state = $rosterOrder.get()
      expect(state[scopeA]).toEqual(['bot-2', 'bot-3'])
      // Section B does not have bot-1 in its order yet
      expect(state[scopeB]).toEqual(['bot-4', 'bot-5'])
    })

    it('prunes deleted bots across all section scopes without leaving stale entries', () => {
      const scopeA = rosterSectionScope('sec-A', 'local')
      const scopeB = rosterSectionScope('sec-B', 'local')

      persistRosterOrder({
        [scopeA]: ['live-1', 'deleted-bot', 'live-2'],
        [scopeB]: ['live-3', 'deleted-bot']
      })

      const liveKeys = new Set(['live-1', 'live-2', 'live-3'])
      pruneRosterOrder(liveKeys)

      const state = $rosterOrder.get()
      expect(state[scopeA]).toEqual(['live-1', 'live-2'])
      expect(state[scopeB]).toEqual(['live-3'])
    })

    it('resets section custom order back to default recency sort', () => {
      const scopeA = rosterSectionScope('sec-A', 'local')
      const scopeB = rosterSectionScope('sec-B', 'local')

      persistRosterOrder({
        [scopeA]: ['bot-a2', 'bot-a1'],
        [scopeB]: ['bot-b2', 'bot-b1']
      })

      expect(Boolean($rosterOrder.get()[scopeA]?.length)).toBe(true)
      expect(hasAnyCustomRosterOrder()).toBe(true)

      // Reset only Section A
      clearRosterOrder(scopeA)

      expect(Boolean($rosterOrder.get()[scopeA]?.length)).toBe(false)
      expect($rosterOrder.get()[scopeA]).toBeUndefined()
      // Section B is still customized
      expect(Boolean($rosterOrder.get()[scopeB]?.length)).toBe(true)
      expect($rosterOrder.get()[scopeB]).toEqual(['bot-b2', 'bot-b1'])

      // Global reset
      clearRosterOrder()
      expect(hasAnyCustomRosterOrder()).toBe(false)
      expect($rosterOrder.get()).toEqual({})
    })

    it('preserves persistence across loadRosterOrder', () => {
      const scopeA = rosterSectionScope('sec-A', 'local')
      persistRosterOrder({
        [scopeA]: ['bot-1', 'bot-2']
      })

      // Simulate app reload
      $rosterOrder.set({})
      loadRosterOrder()

      expect($rosterOrder.get()[scopeA]).toEqual(['bot-1', 'bot-2'])
    })

    it('snapshots, clears, and restores section custom order across connections on section Delete + Undo', () => {
      const localA = rosterSectionScope('sec-A', 'local')
      const remoteA = rosterSectionScope('sec-A', 'ssh-box')
      const localB = rosterSectionScope('sec-B', 'local')

      persistRosterOrder({
        [localA]: ['local::bot-1', 'local::bot-2'],
        [remoteA]: ['ssh-box::bot-3', 'ssh-box::bot-4'],
        [localB]: ['local::bot-5']
      })

      const snapshot = snapshotSectionRosterOrder('sec-A')
      expect(snapshot).toEqual({
        [localA]: ['local::bot-1', 'local::bot-2'],
        [remoteA]: ['ssh-box::bot-3', 'ssh-box::bot-4']
      })

      clearSectionRosterOrder('sec-A')
      expect($rosterOrder.get()).toEqual({
        [localB]: ['local::bot-5']
      })

      restoreSectionRosterOrder(snapshot)
      expect($rosterOrder.get()).toEqual({
        [localA]: ['local::bot-1', 'local::bot-2'],
        [remoteA]: ['ssh-box::bot-3', 'ssh-box::bot-4'],
        [localB]: ['local::bot-5']
      })
    })

    it('removes a moved bot from its previous section across connection prefixes without touching destination order', () => {
      const localA = rosterSectionScope('sec-A', 'local')
      const legacyA = rosterSectionScope('sec-A', 'legacy')
      const localB = rosterSectionScope('sec-B', 'local')

      persistRosterOrder({
        [localA]: ['legacy::alpha', 'legacy::beta'],
        [legacyA]: ['legacy::alpha', 'legacy::gamma'],
        [localB]: ['legacy::delta', 'legacy::alpha']
      })

      removeBotFromSectionOrder({ name: 'alpha' }, 'sec-A')

      expect($rosterOrder.get()).toEqual({
        [localA]: ['legacy::beta'],
        [legacyA]: ['legacy::gamma'],
        [localB]: ['legacy::delta', 'legacy::alpha']
      })
    })
  })

  describe('composition with #113414 group chats and #149444 connect-on-demand gateways', () => {
    it('preserves pinned outer bands and group-chat slot positions in orderSectionMixedRows', () => {
      type MixedRow =
        | { activity: number; bot: RosterRow; created: number; kind: 'bot'; pinned: boolean }
        | { kind: 'group'; name: string; pinned: boolean }

      const mixed: MixedRow[] = [
        { kind: 'bot', bot: { connectionId: 'local', name: 'pin-bot' }, pinned: true, activity: 50, created: 1 },
        { kind: 'group', name: 'Pinned Crew', pinned: true },
        { kind: 'bot', bot: { connectionId: 'local', name: 'bot-a' }, pinned: false, activity: 300, created: 2 },
        { kind: 'group', name: 'Unpinned Crew', pinned: false },
        { kind: 'bot', bot: { connectionId: 'local', name: 'bot-b' }, pinned: false, activity: 200, created: 3 }
      ]

      // Empty orderKeys returns the original array untouched.
      expect(orderSectionMixedRows(mixed, row => row, [])).toBe(mixed)

      // Custom order swaps bot-b before bot-a while keeping Pinned Crew in the pinned band
      // and Unpinned Crew in its exact unpinned slot (never above pin-bot).
      const ordered = orderSectionMixedRows(mixed, row => row, ['local::bot-b', 'local::bot-a'])

      expect(ordered.map(row => (row.kind === 'group' ? `group:${row.name}` : `bot:${row.bot.name}`))).toEqual([
        'bot:pin-bot',
        'group:Pinned Crew',
        'bot:bot-b',
        'group:Unpinned Crew',
        'bot:bot-a'
      ])
    })

    it('preserves custom order for offline, errored, and SSH connect-on-demand gateways during pruneRosterOrder', () => {
      const localScope = rosterSectionScope('sec-A', 'local')
      const sshScope = rosterSectionScope('sec-A', 'ssh-prod')
      const errScope = rosterSectionScope('sec-A', 'gateway-err')
      const downScope = rosterSectionScope('sec-A', 'cloud-down')
      const removedScope = rosterSectionScope('sec-A', 'old-removed')

      persistRosterOrder({
        [localScope]: ['local::keep', 'local::deleted'],
        [sshScope]: ['ssh-prod::deferred-1', 'ssh-prod::deferred-2'],
        [errScope]: ['gateway-err::flapping-1'],
        [downScope]: ['cloud-down::offline-1'],
        [removedScope]: ['old-removed::stale-1']
      })

      const sources: GatewaySource[] = [
        { connectionId: 'local', reachable: true },
        { connectionId: 'ssh-prod', error: 'connect-on-demand', reachable: false },
        { connectionId: 'gateway-err', error: 'ECONNRESET', reachable: true },
        { connectionId: 'cloud-down', reachable: false }
      ]

      pruneRosterOrder(new Set(['local::keep']), sources)

      expect($rosterOrder.get()).toEqual({
        [localScope]: ['local::keep'],
        [sshScope]: ['ssh-prod::deferred-1', 'ssh-prod::deferred-2'],
        [errScope]: ['gateway-err::flapping-1'],
        [downScope]: ['cloud-down::offline-1']
      })

      // Unknown source inventory (sources === undefined) preserves remote connection scopes
      // while pruning deleted local/legacy keys.
      persistRosterOrder({
        [localScope]: ['local::keep', 'local::gone'],
        [sshScope]: ['ssh-prod::deferred-1']
      })

      pruneRosterOrder(new Set(['local::keep']), undefined)

      expect($rosterOrder.get()).toEqual({
        [localScope]: ['local::keep'],
        [sshScope]: ['ssh-prod::deferred-1']
      })
    })
  })
})
