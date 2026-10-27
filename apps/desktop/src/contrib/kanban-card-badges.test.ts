/**
 * The Kanban card-badge contract (#140026): plugins contribute small text
 * badges per card through `KANBAN_CARD_AREAS.badges`; the host validates,
 * orders, caps and paints them. These drive the REAL plugin context and
 * registry: what a plugin can register, and what the host will accept.
 */
import {
  KANBAN_CARD_AREAS,
  KANBAN_CARD_BADGE_HUES,
  KANBAN_CARD_BADGE_LIMITS,
  type KanbanCard,
  type KanbanCardBadgeContribution,
  type KanbanCardBoard
} from '@hermes/plugin-sdk'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { resolveKanbanCardBadges } from './kanban-card-badges'
import { createPluginContext } from './plugin'
import { registry } from './registry'

const card: KanbanCard = { assignee: 'alice', id: 't_1', status: 'ready', tenant: 'acme' }

const loaded: Array<() => void> = []

/** Load a plugin the way the loader does: its disposers are collected and run
 *  together on unload/disable. Returns that unload. */
function loadPlugin(id: string, contributions: Record<string, KanbanCardBadgeContribution['decorate']>) {
  const disposers: Array<() => void> = []
  const ctx = createPluginContext(id, dispose => disposers.push(dispose))

  for (const [localId, decorate] of Object.entries(contributions)) {
    ctx.register({ area: KANBAN_CARD_AREAS.badges, data: { decorate }, id: localId })
  }

  const unload = () => disposers.splice(0).forEach(dispose => dispose())
  loaded.push(unload)

  return unload
}

const badgesFor = (target: KanbanCard = card, board: KanbanCardBoard = { tenants: [] }) =>
  resolveKanbanCardBadges(registry.getArea(KANBAN_CARD_AREAS.badges), target, board)

afterEach(() => {
  loaded.splice(0).forEach(unload => unload())
  vi.restoreAllMocks()
})

describe('kanban card badges: ordering and arbitration', () => {
  it('orders badges by plugin id, not by load order or a plugin-chosen `order`', () => {
    const disposers: Array<() => void> = []
    const zeta = createPluginContext('zeta', d => disposers.push(d))
    const alpha = createPluginContext('alpha', d => disposers.push(d))
    loaded.push(() => disposers.splice(0).forEach(d => d()))

    // zeta loads first AND bids for the front of the row; neither moves it.
    zeta.register({ area: KANBAN_CARD_AREAS.badges, data: { decorate: () => ({ label: 'Z' }) }, id: 'b', order: -1e9 })
    alpha.register({ area: KANBAN_CARD_AREAS.badges, data: { decorate: () => ({ label: 'A' }) }, id: 'b' })

    expect(badgesFor().map(b => b.label)).toEqual(['A', 'Z'])
  })

  it('caps each plugin across all of its contributions, without starving other plugins', () => {
    loadPlugin('noisy', {
      one: () => [{ label: 'n1' }, { label: 'n2' }],
      two: () => [{ label: 'n3' }, { label: 'n4' }]
    })
    loadPlugin('quiet', { tag: () => ({ label: 'q1' }) })

    const badges = badgesFor()

    expect(badges.filter(b => b.source === 'plugin:noisy')).toHaveLength(KANBAN_CARD_BADGE_LIMITS.perPlugin)
    expect(badges.map(b => b.label)).toContain('q1')
  })

  it('a decorator that throws declines; the rest of the row still paints', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    loadPlugin('broken', {
      bad: () => {
        throw new Error('boom')
      }
    })
    loadPlugin('fine', { tag: () => ({ label: 'ok' }) })

    expect(badgesFor().map(b => b.label)).toEqual(['ok'])
  })

  it('passes each decorator a frozen, minimal view of the card and its board', () => {
    const seen: unknown[] = []
    loadPlugin('peek', {
      tag: (c, board) => {
        seen.push(c, board)

        return null
      }
    })

    badgesFor({ ...card, body: 'more prose', title: 'secret prose' } as unknown as KanbanCard, {
      tenants: ['acme', 'globex']
    })

    expect(seen).toEqual([card, { tenants: ['acme', 'globex'] }])
    expect(seen.every(view => Object.isFrozen(view))).toBe(true)
    expect(Object.isFrozen((seen[1] as { tenants: string[] }).tenants)).toBe(true)
  })
})

describe('kanban card badges: what a plugin can put on a card', () => {
  it('keeps text only: trims, strips control and bidi characters, and caps the length', () => {
    loadPlugin('text', {
      tag: () => [{ label: `  \u202Eevil\u0007 ${'x'.repeat(80)}  `, tooltip: `tip\u0000 ${'y'.repeat(400)}` }]
    })

    const [badge] = badgesFor()

    expect(badge.label.startsWith('evil x')).toBe(true)
    expect([...badge.label].length).toBeLessThanOrEqual(KANBAN_CARD_BADGE_LIMITS.labelChars)
    expect(['\u202E', '\u0007'].some(ch => badge.label.includes(ch))).toBe(false)
    expect([...(badge.tooltip ?? '')].length).toBeLessThanOrEqual(KANBAN_CARD_BADGE_LIMITS.tooltipChars)
    expect(badge.tooltip?.includes('\u0000')).toBe(false)
  })

  it('accepts only palette tones; anything else paints neutral', () => {
    loadPlugin('tones', {
      tag: () => [
        { label: 'hue', tone: KANBAN_CARD_BADGE_HUES[3] },
        { label: 'css', tone: 'red; background:url(x)' as never }
      ]
    })

    expect(badgesFor().map(b => b.tone)).toEqual([KANBAN_CARD_BADGE_HUES[3], 'neutral'])
  })

  it('ignores non-text payloads: no label, a non-string label, or extra render fields', () => {
    loadPlugin('junk', {
      tag: () =>
        [
          { label: '' },
          { label: { toString: () => 'obj' } },
          { html: '<b>x</b>', label: 'kept', render: () => 'node' },
          'bare string',
          null
        ] as never
    })

    const badges = badgesFor()

    expect(badges.map(b => b.label)).toEqual(['kept'])
    expect(Object.keys(badges[0]).sort()).toEqual(['key', 'label', 'source', 'tone', 'tooltip'])
  })
})

describe('kanban card badges: teardown', () => {
  it('unloading a plugin removes every badge it contributed and leaves the others', () => {
    const unloadA = loadPlugin('a', { one: () => ({ label: 'from a' }), two: () => ({ label: 'also a' }) })
    loadPlugin('b', { one: () => ({ label: 'from b' }) })

    expect(badgesFor().map(b => b.label)).toEqual(['from a', 'also a', 'from b'])

    unloadA()

    expect(badgesFor().map(b => b.label)).toEqual(['from b'])
  })
})
