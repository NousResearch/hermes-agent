/**
 * Bot Chat tab identity across a compression tip rotation the window did not
 * stream.
 *
 * A Bot Chat turn that runs outside this window (a teammate's
 * `message_agent`, a cron `bot-chat:` delivery, a CLI turn) can auto-compress
 * and rotate the conversation's stored id. Nothing re-keys the open tab — the
 * live rotation path (#98622) only fires for a runtime this window streams —
 * so the tab stays keyed to the old segment until the next registry
 * resolution notices.
 *
 * That resolution is a BACKGROUND wake (`session.reclaimed`, roster
 * activity). It used to DISCARD the old-segment tab as stale (#120810) and
 * then refresh the tip in place (#121874) — which opens nothing when the tip
 * has no tab. The Bot Chat tab vanished and the strip fell back to whatever
 * tab sat beside it: an old, unrelated session in the Bot Chat's place.
 *
 * Real stores, real layout tree, real pane mirror — the swap only exists in
 * the composition.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { findGroupOfPane } from '@/components/pane-shell/tree/model'

const tilePane = (id: string) => `session-tile:${id}`
const OWNER = 'bot:ops'
const BOT_CHAT = 'Bot Chat'

/** The probe Bot Mode passes: a canonical-titled tab keyed to none of the ids
 *  the registry currently resolves (canonical-chat.ts isStaleBotChatTile). */
const staleBotChat =
  (canonicalIds: readonly string[]) => (tile: { storedSessionId: string; workspaceTabTitle?: string }) =>
    tile.workspaceTabTitle === BOT_CHAT && !canonicalIds.includes(tile.storedSessionId)

describe('Bot Chat tab survives a tip rotation resolved by a background wake', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    vi.resetModules()
  })

  async function setup() {
    const tree = await import('@/components/pane-shell/tree/store')
    const model = await import('@/components/pane-shell/tree/model')
    const { registry } = await import('@/contrib/registry')
    const session = await import('@/store/session')
    const states = await import('@/store/session-states')

    registry.register({
      area: 'panes',
      data: { placement: 'main', uncloseable: true },
      id: 'workspace',
      render: () => null,
      title: 'chat'
    })

    const { paneMirror } = await import('@/app/chat/pane-mirror')
    paneMirror({
      source: states.$sessionTiles,
      key: tile => tile.storedSessionId,
      prefix: 'session-tile',
      dir: tile => tile.dir,
      anchor: tile => tile.anchor,
      before: tile => tile.before,
      minWidth: '10rem',
      title: id => id,
      render: () => null,
      close: states.closeSessionTile
    })()
    tree.watchContributedPanes()
    session.$selectedStoredSessionId.set('primary')
    tree.declareDefaultTree(model.group(['workspace'], { active: 'workspace', id: 'grp-main' }))

    const botScope = { workspaceMode: 'bots' as const, workspaceOwnerKey: OWNER }

    // The strip from the report: an old session tab left open under the bot, the Bot
    // Chat (keyed to the segment that was the tip when it opened) in front of
    // it, and another tab after.
    states.openSessionTile('old-session', 'center', 'workspace', undefined, botScope)
    states.openSessionTile('bot-chat-seg-2', 'center', 'workspace', undefined, {
      ...botScope,
      workspaceTabTitle: BOT_CHAT
    })
    states.openSessionTile('side-thread', 'center', 'workspace', undefined, botScope)
    states.focusOpenSession('bot-chat-seg-2', botScope)

    const strip = () => findGroupOfPane(tree.$layoutTree.get()!, 'workspace')!

    expect(strip().panes).toEqual([
      'workspace',
      tilePane('old-session'),
      tilePane('bot-chat-seg-2'),
      tilePane('side-thread')
    ])
    expect(strip().active).toBe(tilePane('bot-chat-seg-2'))

    return { states, strip }
  }

  it('re-homes the open Bot Chat tab onto the new tip instead of dropping it', async () => {
    const { states, strip } = await setup()

    // Registry now resolves root-1 → tip seg-3; the wake reconciles the
    // owner's tabs against it without fronting anything (`[]`).
    states.focusWorkspaceOwnerSessionTile(OWNER, staleBotChat(['root-1', 'bot-chat-seg-3']), [], 'bot-chat-seg-3')

    const botChatTabs = states.$sessionTiles.get().filter(tile => tile.workspaceTabTitle === BOT_CHAT)

    // One Bot Chat tab, on the live tip, still this bot's.
    expect(botChatTabs.map(tile => tile.storedSessionId)).toEqual(['bot-chat-seg-3'])
    expect(botChatTabs[0]).toMatchObject({ workspaceMode: 'bots', workspaceOwnerKey: OWNER })
    // Same slot, still the tab on screen — not the old session beside it.
    expect(strip().panes).toEqual([
      'workspace',
      tilePane('old-session'),
      tilePane('bot-chat-seg-3'),
      tilePane('side-thread')
    ])
    expect(strip().active).toBe(tilePane('bot-chat-seg-3'))
  })

  it('never fronts the re-homed tab when the user was on another tab', async () => {
    const { states, strip } = await setup()
    states.focusOpenSession('side-thread', { workspaceMode: 'bots', workspaceOwnerKey: OWNER })

    states.focusWorkspaceOwnerSessionTile(OWNER, staleBotChat(['root-1', 'bot-chat-seg-3']), [], 'bot-chat-seg-3')

    expect(strip().panes).toEqual([
      'workspace',
      tilePane('old-session'),
      tilePane('bot-chat-seg-3'),
      tilePane('side-thread')
    ])
    // A background wake must not take the strip (#121874).
    expect(strip().active).toBe(tilePane('side-thread'))
  })

  it('drops the stale binding so the re-homed tab resumes the tip, not the old runtime', async () => {
    const { states } = await setup()
    states.patchSessionTile('bot-chat-seg-2', { runtimeId: 'runtime-old-segment' })

    states.focusWorkspaceOwnerSessionTile(OWNER, staleBotChat(['root-1', 'bot-chat-seg-3']), [], 'bot-chat-seg-3')

    expect(
      states.$sessionTiles.get().find(tile => tile.storedSessionId === 'bot-chat-seg-3')?.runtimeId
    ).toBeUndefined()
  })

  it('keeps one tab when the tip is already open: the stale twin goes, the live one takes its place on screen', async () => {
    const { states, strip } = await setup()
    states.openSessionTile('bot-chat-seg-3', 'center', 'workspace', undefined, {
      workspaceMode: 'bots',
      workspaceOwnerKey: OWNER,
      workspaceTabTitle: BOT_CHAT
    })
    states.focusOpenSession('bot-chat-seg-2', { workspaceMode: 'bots', workspaceOwnerKey: OWNER })

    states.focusWorkspaceOwnerSessionTile(OWNER, staleBotChat(['root-1', 'bot-chat-seg-3']), [], 'bot-chat-seg-3')

    expect(
      states.$sessionTiles
        .get()
        .filter(tile => tile.workspaceTabTitle === BOT_CHAT)
        .map(tile => tile.storedSessionId)
    ).toEqual(['bot-chat-seg-3'])
    expect(strip().panes).not.toContain(tilePane('bot-chat-seg-2'))
    expect(strip().active).toBe(tilePane('bot-chat-seg-3'))
  })

  it('without a re-home target the probe still discards (roster click keeps its #90102 contract)', async () => {
    const { states } = await setup()

    expect(states.focusWorkspaceOwnerSessionTile(OWNER, staleBotChat(['root-1', 'bot-chat-seg-3']), [])).toBeNull()
    expect(states.$sessionTiles.get().some(tile => tile.workspaceTabTitle === BOT_CHAT)).toBe(false)
  })
})
