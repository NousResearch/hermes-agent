/**
 * The row-level reads a roster row is assembled from: who sent the last
 * message, and whether a bot counts as live right now.
 *
 * Two bug classes are pinned:
 *  - #89484 — the bot-to-bot badge rendered the raw captured profile name, so
 *    the primary profile surfaced as @default instead of @hermes;
 *  - the "6d ago" class — canonical Bot Chats are hidden from session lists,
 *    so a bot DM'd all day read as a week idle because its newest VISIBLE
 *    session was a week old. Liveness keys off `botActivitySession`, and
 *    kanban/tool workers count too (hermes-agent#90268): a profile grinding
 *    through a 30-minute task must not read "3 hr ago" the whole time.
 */

import { describe, expect, it, vi } from 'vitest'

import {
  ACTIVE_WINDOW_S,
  activeBots,
  botCanonicalSessionId,
  botRowOwnsWorkspace,
  previewKind,
  rosterActivityMatches,
  workerActiveAt
} from './row-helpers'
import type { RosterRow } from './types'

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return {
    atom,
    host: { state: { connectionId: { get: () => 'local' } } },
    queryClient: undefined,
    useQuery: vi.fn(),
    useValue: vi.fn()
  }
})

vi.mock('./shared', () => ({ getPluginCtx: () => null, ID: 'hermes-bots' }))

/** Gateway rows carry a session id on `last_session` / `worker_session` that
 *  the plugin's `SessionPreview` type deliberately does not model. Fixtures
 *  keep it so "the canonical id is never the last_session id" stays a real
 *  assertion rather than one about a missing field. */
interface RowFixture extends Omit<Partial<RosterRow>, 'last_session' | 'worker_session'> {
  last_session?: { id?: string; last_active?: number; preview?: string }
  worker_session?: { id?: string; last_active?: number; source?: string }
}

const row = (bot: RowFixture) => bot as RosterRow

// Fixed clock so "inside the window" vs "stale" is deterministic.
const NOW = 1_000_000_000_000
const secondsAgo = (n: number) => NOW / 1000 - n

describe('previewKind classifies a roster preview', () => {
  const fromBot = (preview: null | string | undefined) => previewKind(preview).fromBot

  it('reads a plain chat preview as a human exchange, not a DM', () => {
    expect(fromBot('Can you check the vault sync?')).toBeNull()
  })

  it('parses the current 🤖 delivery prefix and the legacy agent shape', () => {
    expect(fromBot('Message from 🤖 manager (@manager): Learn-share: skill installed')).toBe('manager')
    expect(fromBot("Message from agent 'researcher': here is the paper")).toBe('researcher')
  })

  it('surfaces the primary profile as @hermes, never @default (#89484)', () => {
    expect(fromBot("Message from agent 'default': deploy is green")).toBe('hermes')
    expect(fromBot("Message from agent 'ops': deploy is green")).toBe('ops')
  })

  it('treats an empty or absent preview as not a DM', () => {
    expect(fromBot('')).toBeNull()
    expect(fromBot(undefined)).toBeNull()
  })
})

describe('the canonical session id everything core-keyed reads', () => {
  it('prefers the compression-lineage tip over the durable registry id', () => {
    expect(botCanonicalSessionId(row({ canonical_session: { id: 'root', last_active: 1, resolved_id: 'tip' } }))).toBe(
      'tip'
    )
    expect(botCanonicalSessionId(row({ canonical_session: { id: 'root', last_active: 1 } }))).toBe('root')
  })

  it('is never last_session — that is a different conversation entirely', () => {
    expect(botCanonicalSessionId(row({ last_session: { id: 'scratch', last_active: 1 } }))).toBeNull()
  })
})

describe('which bots are working right now', () => {
  const roster = [
    row({ last_session: { id: 'a', last_active: secondsAgo(10) }, name: 'researcher' }),
    row({ last_session: { id: 'b', last_active: secondsAgo(400) }, name: 'scribe' }),
    row({ name: 'analyst' })
  ]

  it('includes the gateway-busy profile before its first response lands', () => {
    // analyst has no session at all — a busy turn must still show it.
    expect(activeBots(roster, 'analyst', 'busy', NOW).map(bot => bot.name)).toContain('analyst')
  })

  it('includes activity inside the liveness window and excludes activity outside it', () => {
    const names = activeBots(roster, 'default', 'open', NOW).map(bot => bot.name)

    expect(names).toContain('researcher')
    expect(names).not.toContain('scribe')
    // Output follows input order and never hides or reorders the full list.
    expect(names).toEqual(['researcher'])
    expect(roster).toHaveLength(3)
  })

  it('returns an empty list when nothing is active, and tolerates no roster', () => {
    expect(activeBots(roster.slice(1), 'default', 'open', NOW)).toEqual([])
    expect(activeBots(null, 'default', 'open', NOW)).toEqual([])
    expect(activeBots([], 'default', 'open', NOW)).toEqual([])
  })

  it('counts Bot Chat activity that last_session cannot see', () => {
    const bots = [
      row({
        canonical_session: { id: 'chat', last_active: secondsAgo(5) },
        last_session: { id: 'old', last_active: secondsAgo(6 * 86_400) },
        name: 'default'
      })
    ]

    expect(activeBots(bots, 'other', 'open', NOW).map(bot => bot.name)).toContain('default')
  })

  it('counts a live kanban/tool worker heartbeat (#90268)', () => {
    // The reported shape: last chat hours ago while the bot is mid-task.
    const working = row({
      last_session: { id: 'chat', last_active: secondsAgo(3 * 3600) },
      name: 'coding',
      worker_session: { id: 'w1', last_active: secondsAgo(30), source: 'kanban' }
    })

    expect(activeBots([working], 'other', 'open', NOW).map(bot => bot.name)).toContain('coding')
    expect(workerActiveAt(working, NOW)).toBe(true)
  })

  it('ignores a finished worker outside the liveness window', () => {
    const finished = row({
      last_session: { id: 'chat', last_active: secondsAgo(3 * 3600) },
      name: 'coding',
      worker_session: { id: 'w1', last_active: secondsAgo(3600), source: 'kanban' }
    })

    expect(activeBots([finished], 'other', 'open', NOW)).toEqual([])
    // Workers get a wider window than chat activity to bridge one missed
    // heartbeat — but not an hour's worth.
    expect(workerActiveAt(finished, NOW)).toBe(false)
    expect(ACTIVE_WINDOW_S).toBeGreaterThan(0)
  })

  it('includes a member whose group turn is running, keyed by source', () => {
    const coder = row({ connectionId: 'local', name: 'coder' })

    expect(activeBots([coder], 'other', 'open', NOW, ['local::coder']).map(bot => bot.name)).toContain('coder')
    // Another connection's same-named member must not light this row.
    expect(activeBots([coder], 'other', 'open', NOW, ['remote::coder'])).toEqual([])
  })
})

describe('the roster activity filter', () => {
  it('passes everything through with no filter', () => {
    expect(rosterActivityMatches({ activity: 0 }, null, NOW)).toBe(true)
    expect(rosterActivityMatches({ activity: 0 }, 'all', NOW)).toBe(true)
  })

  it('splits recent from older on the week boundary', () => {
    const week = 7 * 24 * 60 * 60 * 1000

    expect(rosterActivityMatches({ activity: NOW - week + 1000 }, 'recent', NOW)).toBe(true)
    expect(rosterActivityMatches({ activity: NOW - week - 1000 }, 'recent', NOW)).toBe(false)
    expect(rosterActivityMatches({ activity: NOW - week - 1000 }, 'older', NOW)).toBe(true)
    // A bot with no activity at all counts as older, never recent.
    expect(rosterActivityMatches({}, 'older', NOW)).toBe(true)
  })

  it('reads the live pulse for the active filter', () => {
    expect(rosterActivityMatches({ active: true }, 'active', NOW)).toBe(true)
    expect(rosterActivityMatches({ active: false }, 'active', NOW)).toBe(false)
  })
})

describe('which row owns the workspace highlight', () => {
  const bot = row({ connectionId: 'local', name: 'ops' })

  it('follows the explicit selection while no bot chat has focus', () => {
    expect(botRowOwnsWorkspace(bot, null, false, null, 'local::ops')).toBe(true)
    expect(botRowOwnsWorkspace(bot, null, false, null, 'local::other')).toBe(false)
  })

  it('follows the FOCUSED chat’s owner once a bot chat has focus', () => {
    // Tab/tile focus moves without swapping the socket, so keying this off
    // the gateway's home profile highlighted the wrong bot.
    const focused = { authoritative: true, connectionId: 'local', name: 'ops' }

    expect(botRowOwnsWorkspace(bot, null, true, focused, 'local::other')).toBe(true)
    expect(botRowOwnsWorkspace(bot, null, true, { ...focused, name: 'writer' }, 'local::ops')).toBe(false)
  })

  it('never highlights a bot row while a group room is active', () => {
    expect(botRowOwnsWorkspace(bot, 'team', false, null, 'local::ops')).toBe(false)
  })
})
