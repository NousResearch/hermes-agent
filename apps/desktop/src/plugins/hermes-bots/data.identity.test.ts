/**
 * The identity layer every Bot Mode surface files a bot under: the @handle it
 * is tagged with, the friendly forms a rename makes taggable, the
 * source-qualified keys that keep two same-named bots apart, which session
 * counts as "what this bot is doing", roster search, and the needs-attention
 * badge.
 *
 * Drives the real `data` module (with real `labels` / `routing` underneath);
 * only the SDK is mocked.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $botAttention,
  $botMeta,
  botActivitySession,
  botFriendlyNames,
  botHandle,
  botMentionTag,
  botMetaKey,
  botRosterKey,
  botSelectionKey,
  clearBotAttention,
  filterBots,
  mentionNameForms,
  noteBotAttention,
  preferReachableSameNameRows,
  resolveRosterMentions
} from './data'
import { indexAliasRoutes } from './routing'
import type { RosterRow } from './types'

const { hostMock } = vi.hoisted(() => ({
  hostMock: {
    request: vi.fn(),
    requestProfile: vi.fn(),
    state: { connectionId: { get: vi.fn(() => 'local') }, profile: { get: vi.fn(() => 'default') } }
  }
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom: nanoAtom } = await import('nanostores')

  return {
    atom: nanoAtom,
    host: hostMock,
    queryClient: undefined,
    useQuery: vi.fn(),
    useValue: vi.fn()
  }
})

vi.mock('./shared', () => ({ getPluginCtx: () => null, ID: 'hermes-bots' }))

/** Gateway rows carry a session id on `last_session` that the plugin's
 *  `SessionPreview` type deliberately does not model; the fixtures keep it so
 *  "which session speaks for a bot" can be asserted by id. */
interface RowFixture extends Omit<Partial<RosterRow>, 'last_session'> {
  last_session?: { id?: string; last_active?: number; preview?: string }
}

const row = (bot: RowFixture) => bot as RosterRow

beforeEach(() => {
  vi.clearAllMocks()
  hostMock.state.connectionId.get.mockReturnValue('local')
  indexAliasRoutes([])
  $botMeta.set({})
  $botAttention.set({})
})

describe('the @handle a bot answers to', () => {
  it('presents the primary profile as @hermes — "default" never surfaces in the UI', () => {
    expect(botHandle('default')).toBe('hermes')
    expect(botHandle('ops')).toBe('ops')
  })

  it('uses the primary profile friendly identity when one exists', () => {
    expect(botHandle('default', row({ display_name: 'Miguel Arcanjuz', name: 'default' }))).toBe('miguel-arcanjuz')
    expect(
      botHandle('default', row({ name: 'default', ui_meta: { 'hermes-bots': { title: 'Miguel' } } }))
    ).toBe('miguel')
  })

  it('prefers a precomputed multi-source handle over the bare name', () => {
    expect(botHandle('default', row({ handle: 'default-vera', name: 'default' }))).toBe('default-vera')
  })
})

describe('renamed bots stay taggable', () => {
  // Discord report, Aug 2026: renaming a bot — Bot Mode title or `hermes
  // profile rename` display_name — must change what you can @-tag it with,
  // while the old profile handle keeps resolving.
  it('reduces a friendly name to its slugged and collapsed forms', () => {
    expect(mentionNameForms('Research Buddy')).toEqual(['research-buddy', 'researchbuddy'])
    expect(mentionNameForms('Ops')).toEqual(['ops'])
  })

  it('drops reserved tokens so a rename cannot hijack a built-in tag', () => {
    expect(mentionNameForms('Hermes')).toEqual([])
    expect(mentionNameForms('@everyone')).toEqual([])
    expect(mentionNameForms('')).toEqual([])
  })

  it('inserts the renamed slug, falling back to the profile handle', () => {
    $botMeta.set({ writer: { title: 'Research Buddy' } })

    expect(botMentionTag(row({ name: 'writer' }))).toBe('research-buddy')
    expect(botMentionTag(row({ name: 'ops' }))).toBe('ops')
    expect(botMentionTag(row({ name: 'default' }))).toBe('hermes')
    // display_name (`hermes profile rename`) drives the tag too.
    expect(botMentionTag(row({ display_name: 'Deal Finder', name: 'scout' }))).toBe('deal-finder')
  })

  it('resolves a renamed bot by its friendly tag AND its old handle', () => {
    $botMeta.set({ writer: { title: 'Research Buddy' } })

    const roster = [
      row({ name: 'default' }),
      row({ name: 'writer' }),
      row({ display_name: 'Deal Finder', name: 'scout' })
    ]

    const live = { connectionId: 'local', name: 'default' }
    const names = (text: string) => resolveRosterMentions(text, roster, live).map(bot => bot.name)

    expect(names('hey @research-buddy check this')).toEqual(['writer'])
    expect(names('ping @dealfinder please')).toEqual(['scout'])
    expect(names('hey @writer')).toEqual(['writer'])
  })
})

describe('resolving @mentions against the roster', () => {
  const roster = [
    row({ connectionId: 'local', name: 'default' }),
    row({ connectionId: 'mac-mini', connectionLabel: 'Mac Mini', name: 'dixie', remoteSource: true }),
    row({ connectionId: 'mac-mini', handle: 'bob-mac-mini', name: 'bob', remoteSource: true })
  ]

  it('reaches Connections bots by bare name and by @name-device handle', () => {
    const live = { connectionId: 'local', name: 'default' }

    expect(resolveRosterMentions('@dixie how is disk space?', roster, live).map(bot => bot.name)).toEqual(['dixie'])
    expect(resolveRosterMentions('ping @bob-mac-mini', roster, live).map(bot => bot.name)).toEqual(['bob'])
  })

  it('never treats @hermes in your own chat as a handoff to yourself', () => {
    expect(resolveRosterMentions('@hermes do it', roster, { connectionId: 'local', name: 'default' })).toEqual([])
    // From ANOTHER bot's chat the same tag is a real handoff.
    expect(
      resolveRosterMentions('@hermes do it', roster, { connectionId: 'mac-mini', name: 'dixie' }).map(bot => bot.name)
    ).toEqual(['default'])
  })

  it('ignores tags inside code spans and fences', () => {
    const live = { connectionId: 'local', name: 'default' }

    expect(resolveRosterMentions('run `@dixie` literally', roster, live)).toEqual([])
    expect(resolveRosterMentions('```\n@dixie\n```', roster, live)).toEqual([])
  })

  it('refuses an ambiguous bare name — the device-qualified handle is required', () => {
    const twins = [
      row({ connectionId: 'a', handle: 'ops-a', name: 'ops', remoteSource: true }),
      row({ connectionId: 'b', handle: 'ops-b', name: 'ops', remoteSource: true })
    ]

    expect(resolveRosterMentions('@ops ping', twins, { connectionId: 'local', name: 'default' })).toEqual([])
    expect(
      resolveRosterMentions('@ops-b ping', twins, { connectionId: 'local', name: 'default' }).map(
        bot => bot.connectionId
      )
    ).toEqual(['b'])
  })
})

describe('source-qualified keys', () => {
  it('gives the same profile name on two sources distinct React keys', () => {
    const here = row({ connectionId: 'local', name: 'default' })
    const there = row({ connectionId: 'vera', name: 'default', remoteSource: true, sourceScoped: true })

    expect(botRosterKey(here)).not.toBe(botRosterKey(there))
  })

  it('keeps selection and metadata keys from colliding across connections', () => {
    const scoped = (connectionId: string) =>
      row({
        connectionId,
        name: 'default',
        route: { connectionId, mode: 'remote', profile: 'default', targetProfile: 'default' },
        sourceScoped: true
      })

    expect(botSelectionKey(scoped('a'))).not.toBe(botSelectionKey(scoped('b')))
    expect(botMetaKey(scoped('a'))).toBe('a::default')
    expect(botMetaKey(scoped('b'))).toBe('b::default')
  })

  it('leaves an unscoped legacy row on its bare name', () => {
    expect(botSelectionKey(row({ name: 'ops' }))).toBe('ops')
    expect(botMetaKey(row({ name: 'ops' }))).toBe('ops')
  })
})

describe('which session speaks for a bot', () => {
  // The "6d ago" bug: canonical Bot Chats are hidden from session lists, so
  // last_session alone never sees them — a bot you DM all day read as a week
  // idle because its newest VISIBLE session was a week old.
  const now = 1_000_000_000

  it('prefers the fresher canonical Bot Chat over a stale visible session', () => {
    const bot = row({
      canonical_session: { id: 'bot-chat', last_active: now - 5, preview: 'fresh DM' },
      last_session: { id: 'old-scratch', last_active: now - 6 * 86_400, preview: 'ancient' }
    })

    expect(botActivitySession(bot)?.preview).toBe('fresh DM')
  })

  it('keeps last_session when that is the fresher one', () => {
    const bot = row({
      canonical_session: { id: 'bot-chat', last_active: now - 3600, preview: 'older DM' },
      last_session: { id: 'scratch', last_active: now - 10, preview: 'just now' }
    })

    expect(botActivitySession(bot)?.preview).toBe('just now')
  })

  it('degrades to whichever summary an older gateway sent', () => {
    expect(botActivitySession(row({ last_session: { last_active: 1, preview: 'visible only' } }))?.preview).toBe(
      'visible only'
    )
    expect(
      botActivitySession(row({ canonical_session: { id: 'canonical', last_active: 1, preview: 'chat only' } }))?.preview
    ).toBe('chat only')
    expect(botActivitySession(row({}))).toBeNull()
    expect(botActivitySession(null)).toBeNull()
  })
})

describe('roster search narrows without re-ranking', () => {
  const roster = [
    row({ name: 'agency-audio-designer', title: 'Audio Designer' }),
    row({ name: 'agency-ai-engineer', title: 'AI Engineer' }),
    row({ name: 'default' })
  ]

  const meta = { 'agency-audio-designer': { title: 'Sound Studio' }, default: {} }

  it('matches the visible display name, case-insensitively', () => {
    expect(filterBots(roster, meta, 'SOUND').map(bot => bot.name)).toEqual(['agency-audio-designer'])
  })

  it('matches profile handles and preserves roster order', () => {
    expect(filterBots(roster, meta, 'agency-').map(bot => bot.name)).toEqual([
      'agency-audio-designer',
      'agency-ai-engineer'
    ])
    expect(filterBots(roster, meta, '@hermes').map(bot => bot.name)).toEqual(['default'])
    expect(filterBots(roster, meta, 'default').map(bot => bot.name)).toEqual(['default'])
  })

  it('also matches roles, message previews, and the source device name', () => {
    const richer = [
      row({
        connectionLabel: 'Work Studio',
        description: 'Release quality and compliance',
        last_session: { id: 's1', last_active: 1, preview: 'Checked the deployment checklist' },
        name: 'reviewer'
      }),
      row({ description: 'Editorial support', name: 'writer' })
    ]

    expect(filterBots(richer, {}, 'compliance')[0].name).toBe('reviewer')
    expect(filterBots(richer, {}, 'deployment checklist')[0].name).toBe('reviewer')
    expect(filterBots(richer, {}, 'work studio')[0].name).toBe('reviewer')
  })

  it('returns the existing roster reference for a blank query', () => {
    expect(filterBots(roster, meta, '   ')).toBe(roster)
  })
})

describe('unreachable same-name twins (#92286)', () => {
  // After Desktop moves to the built-in This-device source, the old loopback
  // row (127.0.0.1:port, not listening) still sits beside the live profile
  // and looks like a second agent. Only sidebar tiles are filtered — routing
  // identities, group members and @-mentions still see every row.
  const live = row({ connectionId: 'local', name: 'default', sourceReachable: true, sourceScoped: true })
  const dead = row({ connectionId: 'loopback', name: 'default', sourceReachable: false, sourceScoped: true })

  it('drops a dead twin when a live copy exists', () => {
    expect(preferReachableSameNameRows([live, dead]).map(bot => bot.connectionId)).toEqual(['local'])
  })

  it('keeps two live sources sharing a profile name', () => {
    const other = row({ connectionId: 'vera', name: 'default', sourceReachable: true, sourceScoped: true })

    expect(preferReachableSameNameRows([live, other])).toHaveLength(2)
  })

  it('counts connect-on-demand as reachable', () => {
    const onDemand = row({
      connectionId: 'vera',
      name: 'default',
      sourceError: 'connect-on-demand',
      sourceScoped: true
    })

    expect(preferReachableSameNameRows([live, onDemand])).toHaveLength(2)
  })

  it('keeps an unreachable row when there is no live twin, so a down source still has a tile', () => {
    expect(preferReachableSameNameRows([dead])).toHaveLength(1)
  })

  it('keeps a ghost so a selected-but-offline owner is never replaced by a twin', () => {
    expect(preferReachableSameNameRows([live, { ...dead, ghost: true }])).toHaveLength(2)
  })

  it('does not mutate the input list', () => {
    const input = [live, dead]

    preferReachableSameNameRows(input)
    expect(input).toHaveLength(2)
  })
})

describe('needs-attention badge (#93091 item 3)', () => {
  // Background failures whose class is attention-worthy (auth, quota, missing
  // config, blocked) badge the roster tile; transient failures never do; the
  // bot's next good turn clears it.
  it('passes a typed reason code straight through', () => {
    for (const code of ['agent_blocked', 'missing_config', 'provider_auth_or_access', 'provider_quota_limit']) {
      noteBotAttention('radar', code)
      expect($botAttention.get().radar.reason).toBe(code)
    }
  })

  it('classifies the raw gateway error strings current backends emit', () => {
    const reasonFor = (text: string) => {
      $botAttention.set({})
      noteBotAttention('radar', text)

      return $botAttention.get().radar?.reason ?? null
    }

    // The anthropic 401 shape observed on current main.
    expect(
      reasonFor(
        'Error code: 401 - {"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}'
      )
    ).toBe('provider_auth_or_access')
    expect(reasonFor('No LLM provider configured. Run hermes model to pick one.')).toBe('missing_config')
    expect(reasonFor('No access token found for profile')).toBe('missing_config')
    expect(reasonFor('Your account is out of funds')).toBe('provider_quota_limit')
    expect(reasonFor('quota exceeded for this billing period')).toBe('provider_quota_limit')
    expect(reasonFor('agent is blocked awaiting approval')).toBe('agent_blocked')
  })

  it('never badges a transient class — a retryable error must not stick', () => {
    for (const text of [
      'Rate limit exceeded, retry shortly',
      'Error code: 429 - too many requests',
      '500 Internal Server Error',
      'upstream 503 service unavailable',
      'the model is overloaded, try again',
      'request timed out after 180s',
      'temporarily unavailable',
      '',
      null,
      undefined
    ]) {
      $botAttention.set({})
      noteBotAttention('radar', text)
      expect($botAttention.get()).toEqual({})
    }
  })

  it('keeps the latest failure per bot, independently, and clears on a good turn', () => {
    noteBotAttention('radar', 'Error code: 401 authentication_error')

    const first = $botAttention.get().radar

    expect(first.reason).toBe('provider_auth_or_access')
    expect(first.at).toBeGreaterThan(0)
    expect(first.message).toMatch(/401/)

    noteBotAttention('radar', 'No LLM provider configured')
    expect($botAttention.get().radar.reason).toBe('missing_config')

    noteBotAttention('dixie', 'quota exceeded')
    expect($botAttention.get().dixie.reason).toBe('provider_quota_limit')

    clearBotAttention('radar')
    expect($botAttention.get().radar).toBeUndefined()
    expect($botAttention.get().dixie.reason).toBe('provider_quota_limit')

    // Clearing an unbadged bot is a no-op.
    clearBotAttention('radar')
    clearBotAttention('')
    expect($botAttention.get().dixie.reason).toBe('provider_quota_limit')
  })
})

describe('alias identity reaches the mention layer (#89131)', () => {
  it('keeps @moxie resolving against the hosted row after handoff', () => {
    indexAliasRoutes([{ connectionId: 'cloud-abc', mode: 'remote', profile: 'moxie', targetProfile: 'default' }])
    $botMeta.set({ 'cloud-abc::moxie': { title: 'Moxie' } })

    const hostedRow = row({
      connectionId: 'cloud-abc',
      name: 'default',
      remoteSource: true,
      sourceScoped: true,
      targetProfile: 'default'
    })

    expect(botFriendlyNames(hostedRow).filter(Boolean)).toEqual(['Moxie'])
    expect(botMentionTag(hostedRow)).toBe('moxie')
  })
})
