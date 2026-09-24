/**
 * Per-bot Hide/Unhide.
 *
 * Right-click → Hide Bot persists `hidden: true` in bot meta (local
 * ctx.storage AND server ui_meta via saveBotMeta), hidden bots drop out of the
 * roster list, and a header eye toggle — rendered only while at least one bot
 * is hidden — reveals them dimmed for right-click → Unhide.
 *
 * Hiding is a roster-DISPLAY concern only: the bot keeps working, stays
 * mentionable, keeps its group membership, and any open chat stays open. The
 * one behavioral consequence is quiet: a hidden bot still accumulates unread,
 * it just never toasts.
 *
 * Ported from tests/hide-bots.test.mjs, which ran the whole plugin.js bundle
 * under `vm`.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

const { markSessionUnreadFinished, notify, request } = vi.hoisted(() => ({
  markSessionUnreadFinished: vi.fn(),
  notify: vi.fn(),
  request: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    ackStoredSessionId: vi.fn(),
    host: { ...sdk.host, notify, request },
    markSessionUnreadFinished
  }
})

/** Every ctx.storage write the module graph made, oldest first. */
interface StorageWrite {
  key: string
  value: Record<string, { hidden?: boolean; title?: string }>
}

/** Bot Mode keeps roster/meta/watermark state in module-level atoms, so each
 *  test gets a fresh graph rather than unwinding the previous one's writes. */
async function loadModules() {
  vi.resetModules()

  const writes: StorageWrite[] = []

  const [botState, data, hiddenBots, profileOps, rosterActions, shared] = await Promise.all([
    import('./bot-state'),
    import('./data'),
    import('./hidden-bots'),
    import('./profile-ops'),
    import('./roster-actions'),
    import('./shared')
  ])

  shared.setPluginCtx({
    storage: {
      get: () => null,
      remove: () => undefined,
      set: (key: string, value: unknown) => {
        writes.push({ key, value } as StorageWrite)
      }
    }
  } as unknown as Parameters<typeof shared.setPluginCtx>[0])

  return { botState, data, hiddenBots, profileOps, rosterActions, writes }
}

// Pay the graph's cold transform once, up front. `loadModules` re-imports on
// every test, and charging that one-time cost to whichever test happens to run
// first makes it time out under a loaded runner.
beforeAll(async () => {
  await loadModules()
}, 60_000)

beforeEach(() => {
  vi.clearAllMocks()
  request.mockResolvedValue({ applied: { ui_meta: true } })
})

describe('hiding persists locally and cross-machine', () => {
  it('writes hidden:true to storage and ships it in the server ui_meta', async () => {
    const { data, writes } = await loadModules()

    await data.saveBotMeta('default', { hidden: true })

    expect(data.$botMeta.get().default.hidden).toBe(true)
    expect(writes.at(-1)?.key).toBe('bot-meta')
    expect(writes.at(-1)?.value.default.hidden).toBe(true)

    const configure = request.mock.calls.find(([method]) => method === 'profiles.configure')

    expect(configure?.[1].ui_meta['hermes-bots'].hidden).toBe(true)
  })

  it('clears the flag on unhide, and the server false beats a stale local true', async () => {
    const { data, profileOps } = await loadModules()

    data.$botMeta.set({ ghost: { hidden: true, title: 'Ghost' } })
    await data.saveBotMeta('ghost', { hidden: false })

    expect(data.$botMeta.get().ghost.hidden).toBe(false)

    const configure = request.mock.calls.find(([method]) => method === 'profiles.configure')

    // The server copy carries the literal false, not an omission.
    expect(configure?.[1].ui_meta['hermes-bots'].hidden).toBe(false)

    // Machine B: its stale local copy still says hidden:true; the server
    // overlay (which merges OVER local) must win with the false.
    data.$botMeta.set({ ghost: { hidden: true, title: 'Ghost' } })
    profileOps.mergeServerMeta([
      { name: 'ghost', ui_meta: { 'hermes-bots': { hidden: false, title: 'Ghost' } } }
    ] as unknown as RosterRow[])

    expect(data.$botMeta.get().ghost.hidden).toBe(false)
  })

  it('lands a hide done on another machine and persists it to storage', async () => {
    const { data, profileOps, writes } = await loadModules()

    data.$botMeta.set({ ghost: { title: 'Ghost' } })
    profileOps.mergeServerMeta([
      { name: 'ghost', ui_meta: { 'hermes-bots': { hidden: true, title: 'Ghost' } } }
    ] as unknown as RosterRow[])

    expect(data.$botMeta.get().ghost.hidden).toBe(true)
    expect(writes.at(-1)?.value.ghost.hidden).toBe(true)
  })
})

describe('the roster drops hidden rows without touching their twins', () => {
  it('keeps a same-named remote-source row visible when the local one is hidden', async () => {
    const { data, hiddenBots } = await loadModules()

    data.$botMeta.set({ ghost: { hidden: true } })

    const roster = [
      { name: 'default' },
      { name: 'ghost' },
      { connectionId: 'mini', name: 'ghost', remoteSource: true }
    ] as RosterRow[]

    const meta = data.$botMeta.get()
    const visible = roster.filter(bot => !hiddenBots.isBotHidden(bot, meta))

    expect(visible.map(bot => `${bot.remoteSource ? 'r:' : ''}${bot.name}`)).toEqual(['default', 'r:ghost'])
  })
})

describe('hiding the selected bot re-homes the selection', () => {
  it('falls back to the first visible bot', async () => {
    const { botState, data, hiddenBots } = await loadModules()

    botState.$selectedBot.set('ghost')
    data.$botMeta.set({ ghost: { hidden: true } })
    data.$lastRoster.set([{ name: 'ghost' }, { name: 'scribe' }, { name: 'default' }] as RosterRow[])

    hiddenBots.fallbackSelectionAfterHide('ghost')

    expect(botState.$selectedBot.get()).toBe('scribe')
  })

  it('falls back to default when nothing else is visible', async () => {
    const { botState, data, hiddenBots } = await loadModules()

    botState.$selectedBot.set('ghost')
    data.$botMeta.set({ ghost: { hidden: true } })
    data.$lastRoster.set([{ name: 'ghost' }] as RosterRow[])

    hiddenBots.fallbackSelectionAfterHide('ghost')

    expect(botState.$selectedBot.get()).toBe('default')
  })

  it('keeps the selection when default itself is hidden and alone', async () => {
    // Routines is scoped to the selected bot — it must not chase a ghost.
    const { botState, data, hiddenBots } = await loadModules()

    botState.$selectedBot.set('default')
    data.$botMeta.set({ default: { hidden: true } })
    data.$lastRoster.set([{ name: 'default' }] as RosterRow[])

    hiddenBots.fallbackSelectionAfterHide('default')

    expect(botState.$selectedBot.get()).toBe('default')
  })

  it('never moves the selection when an unselected bot is hidden', async () => {
    const { botState, data, hiddenBots } = await loadModules()

    botState.$selectedBot.set('scribe')
    data.$botMeta.set({ ghost: { hidden: true } })
    data.$lastRoster.set([{ name: 'ghost' }, { name: 'scribe' }] as RosterRow[])

    hiddenBots.fallbackSelectionAfterHide('ghost')

    expect(botState.$selectedBot.get()).toBe('scribe')
  })
})

describe('a hidden bot stays quiet without going deaf', () => {
  it('accumulates unread but never toasts, even with toasts on', async () => {
    const { botState, data, rosterActions } = await loadModules()

    rosterActions.setActivityToasts(true)
    data.$botMeta.set({ ghost: { hidden: true } })
    botState.$selectedBot.set('default')

    const rosterAt = (at: number) =>
      [
        {
          canonical_session: { id: 'ghost-chat', last_active: at, preview: 'Message from writer: hi' },
          name: 'ghost'
        }
      ] as unknown as RosterRow[]

    rosterActions.trackInboundActivity(rosterAt(100)) // seeding poll
    rosterActions.trackInboundActivity(rosterAt(200)) // activity past the watermark

    // Unread goes straight into core's store, keyed by the same canonical id
    // the row's status dot reads.
    expect(markSessionUnreadFinished).toHaveBeenCalledWith('ghost-chat', 'ghost')
    expect(notify).not.toHaveBeenCalled()
  })
})
