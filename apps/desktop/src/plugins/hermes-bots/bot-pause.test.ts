import { host, queryClient } from '@hermes/plugin-sdk'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { botForSession, resolveBotNotificationLevel, revertCachedPause, toggleBotPaused } from './bot-pause'
import { $botMeta, $lastRoster, BOT_META_V1_KEY, botOwner, ROSTER_KEY } from './data'
import { setPluginCtx } from './shared'
import type { RosterRow } from './types'

const roster: RosterRow[] = [
  { name: 'default', canonical_session: { id: 'chat-local' } },
  { name: 'default', connectionId: 'cloud', canonical_session: { id: 'chat-cloud' } },
  { name: 'scribe', canonical_session: { id: 'chat-scribe', resolved_id: 'chat-scribe-tip' } }
]

describe('botForSession', () => {
  it('two same-named bots on different connections never answer for each other', () => {
    expect(botForSession(roster, { owner: { connectionId: 'cloud', profile: 'default' }, storedSessionId: 'x' }, 'local'))
      .toBe(roster[1])
    expect(botForSession(roster, { owner: { connectionId: null, profile: 'default' }, storedSessionId: 'x' }, 'local'))
      .toBe(roster[0])
    // An owner on a connection this roster has no row for is nobody's.
    expect(botForSession(roster, { owner: { connectionId: 'laptop', profile: 'default' }, storedSessionId: 'x' }, 'local'))
      .toBeUndefined()
  })

  it('an alias owner matches its backend row once the roster shows that row instead', () => {
    const backend: RosterRow[] = [{ name: 'writer', connectionId: 'cloud', canonical_session: { id: 'w' } }]
    const owner = { connectionId: 'cloud', profile: 'scribe-alias', targetProfile: 'writer' }

    expect(botForSession(backend, { owner, storedSessionId: 'x' }, 'local')).toBe(backend[0])
  })

  it("falls back to the bot whose canonical Bot Chat the session is (durable id or lineage tip)", () => {
    expect(botForSession(roster, { owner: null, storedSessionId: 'chat-scribe-tip' }, 'local')).toBe(roster[2])
    expect(botForSession(roster, { owner: null, storedSessionId: 'unrelated' }, 'local')).toBeUndefined()
  })
})

describe('revertCachedPause', () => {
  const bot: RosterRow = { name: 'scribe' }

  it('restores only the value this save wrote, never a newer one', () => {
    const { key } = botOwner(bot)

    $botMeta.set({ [key]: { paused: true } })
    revertCachedPause(bot, true, false)
    expect($botMeta.get()[key]?.paused).toBe(false)

    $botMeta.set({ [key]: { paused: true } }) // another window paused it since: that stays
    revertCachedPause(bot, false, true)
    expect($botMeta.get()[key]?.paused).toBe(true)
  })
})

describe('a failed pause after a restart', () => {
  afterEach(() => setPluginCtx(null))

  it('the rollback is saved to local storage too, so a relaunch does not load the failed pause', async () => {
    const set = vi.fn()

    setPluginCtx({ storage: { get: vi.fn(), remove: vi.fn(), set } } as never)
    const bot: RosterRow = { name: 'scribe' }
    const { key } = botOwner(bot)

    $botMeta.set({ [key]: { paused: true } })
    await revertCachedPause(bot, true, false)

    expect(set).toHaveBeenCalledWith(BOT_META_V1_KEY, { [key]: { paused: false } })
  })
})

describe('resolveBotNotificationLevel', () => {
  afterEach(() => queryClient.removeQueries({ queryKey: ROSTER_KEY }))

  it("applies a bot's level before the Bots pane ever mounted (roster cache primed at gateway open)", () => {
    $lastRoster.set([])
    $botMeta.set({ scribe: { notifications: 'off' } })
    queryClient.setQueryData([...ROSTER_KEY, 'local'], { profiles: [{ name: 'scribe' }] })

    expect(resolveBotNotificationLevel({ owner: { connectionId: null, profile: 'scribe' }, storedSessionId: 'x' }))
      .toBe('off')
  })
})

describe('toggleBotPaused on an older gateway', () => {
  afterEach(() => vi.restoreAllMocks())

  it('one that stores the flag without enforcing it shows update-required and takes the pause back off', async () => {
    const stored: { paused?: boolean } = {}
    const writes: unknown[] = []

    vi.spyOn(host, 'request').mockImplementation((async (method: string, params: Record<string, unknown>) => {
      if (method === 'profiles.list') {
        return { profiles: [{ name: 'scribe', ui_meta: { 'hermes-bots': { ...stored } } }] }
      }

      const meta = (params.ui_meta as Record<string, { paused?: boolean }>)['hermes-bots']

      writes.push(meta.paused)
      Object.assign(stored, meta)

      return { applied: { ui_meta: true } } // no bot_pause_enforced: this build predates pause
    }) as never)
    const notify = vi.spyOn(host, 'notify').mockImplementation(() => undefined as never)
    const bot: RosterRow = { name: 'scribe' }

    $botMeta.set({})
    await toggleBotPaused(bot)

    expect(writes).toEqual([true, false])
    expect($botMeta.get()[botOwner(bot).key]?.paused).toBe(false)
    expect(notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'error', message: expect.stringMatching(/update/i) }))
  })
})
