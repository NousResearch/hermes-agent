/**
 * The /new → /compact guard's identity check.
 *
 * `/new` inside a bot's canonical forever-chat would fork the relationship
 * into a scratch session — the one thing Bots mode promises never happens. The
 * composer middleware reroutes it to `/compact` (same felt effect: fresh
 * working context, SAME conversation), but only when the chat on screen really
 * is the forever-chat.
 *
 * Identity is the NAME — the profile's session titled "Bot Chat" — reported by
 * the gateway as `canonical_session` on every roster row. No stored meta.chat
 * pointer is consulted: pointers dangle, the registry row cannot.
 *
 * Its predecessor could only regex the guard's source text for the strings
 * `canonical_session` / `canonical?.id`, which is precisely why it never
 * noticed the guard was dead in production: the id it compared against came
 * from `host.activeSessionId`, a property that does not exist. The comparison
 * ran against null on every turn, so `/new` reset forever-chats for as long as
 * the guard shipped. These tests drive the comparison itself.
 */

import { describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

vi.mock('@hermes/plugin-sdk', () => ({
  BOT_CHAT_SESSION_HYDRATION_TIMEOUT_MS: 15_000,
  host: {}
}))
vi.mock('./routing', () => ({
  backendTargetProfile: (_route: unknown, name: string) => name,
  botConnectionRoute: () => null,
  botRosterMeta: () => ({}),
  botWorkspaceOwnerKey: () => '',
  requestForBot: vi.fn()
}))
vi.mock('./data', () => ({
  $botMeta: { get: () => ({}), set: vi.fn() },
  botMetaKey: () => '',
  botOwner: (owner: string) => ({ bot: { name: owner }, key: owner, name: owner, route: null }),
  persistBotMetaSnapshot: vi.fn()
}))
vi.mock('./shared', () => ({ getPluginCtx: () => null }))

const { isCanonicalChatOnScreen } = await import('./canonical-chat')

function bot(canonical: null | { id?: string; resolved_id?: string }): RosterRow {
  return { canonical_session: canonical, name: 'ops' } as RosterRow
}

describe('isCanonicalChatOnScreen', () => {
  it('matches the durable registry row', () => {
    expect(isCanonicalChatOnScreen(bot({ id: 'forever-chat' }), 'forever-chat')).toBe(true)
  })

  it('matches the compression-lineage tip', () => {
    // A compacted Bot Chat is on screen under its tip id while the registry
    // still names it by the root — both are the same forever-chat.
    const compacted = bot({ id: 'root-1', resolved_id: 'tip-9' })

    expect(isCanonicalChatOnScreen(compacted, 'tip-9')).toBe(true)
    expect(isCanonicalChatOnScreen(compacted, 'root-1')).toBe(true)
  })

  it('leaves an ordinary session alone', () => {
    // Sessions-mode scratchpads on the same profile keep full /new freedom.
    expect(isCanonicalChatOnScreen(bot({ id: 'forever-chat' }), 'scratch')).toBe(false)
  })

  it('declines when the bot has no registry row yet', () => {
    expect(isCanonicalChatOnScreen(bot(null), 'anything')).toBe(false)
  })

  it('declines on a missing id rather than matching everything', () => {
    // The regression itself: a null id must never satisfy the guard, and must
    // never be read as "no canonical chat, so let /new through" by accident of
    // a loose comparison.
    expect(isCanonicalChatOnScreen(bot({ id: 'forever-chat' }), null)).toBe(false)
    expect(isCanonicalChatOnScreen(bot({ id: 'forever-chat' }), '')).toBe(false)
    expect(isCanonicalChatOnScreen(undefined, 'forever-chat')).toBe(false)
  })
})
