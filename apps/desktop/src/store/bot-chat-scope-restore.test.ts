/**
 * A bot chat usually lives in MAIN, which has no tile to carry its bot-mode
 * scope, so the scope is kept per stored id beside the tiles. It must survive
 * a reload the way tile scope does: otherwise the main tab of a restored Bot
 * Chat reads "Bot Chat" once the roster labels are published, and keeps
 * reading it until the user re-opens the chat from its bot row.
 *
 * Each test "relaunches" with vi.resetModules() over the same localStorage.
 */

// Loaded once at collection, outside every test's budget: vi.resetModules()
// keeps each module's transformed code, so a relaunch only re-evaluates it.
import '@/components/pane-shell/workspace-scope'
import '@/lib/storage'
import '@/store/session-states'

import { beforeEach, describe, expect, it, vi } from 'vitest'

const CANONICAL_TITLE = 'Bot Chat'
const STORED_ID = '20260825_220141_9f047e'
const CHIEF_KEY = 'bot:local::tranquillity-chief'
const CHIEF_ROUTE = { connectionId: 'local', profile: 'tranquillity-chief', targetProfile: 'tranquillity-chief' }

const chiefScope = {
  ownerRoute: CHIEF_ROUTE,
  workspaceMode: 'bots' as const,
  workspaceOwnerKey: CHIEF_KEY,
  workspaceTabTitle: CANONICAL_TITLE
}

async function launch() {
  vi.resetModules()
  const states = await import('./session-states')
  const scope = await import('@/components/pane-shell/workspace-scope')

  return {
    ...states,
    setWorkspaceOwnerLabel: scope.setWorkspaceOwnerLabel,
    // The main tab's caption, as app/contrib/controller.tsx syncWorkspaceTitle
    // derives it for the selected stored session.
    mainTabCaption: (storedId: string, storedTitle: string) =>
      scope.workspaceOwnerTitle(storedTitle, states.$botChatScopes.get()[storedId])
  }
}

beforeEach(() => {
  window.localStorage.clear()
})

describe('a bot chat restored into the main tab', () => {
  it('captions with its bot once the labels are published, without being re-opened', async () => {
    const before = await launch()
    before.setSessionTileWorkspaceScope(STORED_ID, chiefScope)
    // Opened in place: main carries it, no tile does.
    expect(before.$sessionTiles.get().some(tile => tile.storedSessionId === STORED_ID)).toBe(false)

    const after = await launch()
    expect(after.mainTabCaption(STORED_ID, CANONICAL_TITLE)).toBe(CANONICAL_TITLE)

    after.setWorkspaceOwnerLabel(CHIEF_KEY, 'Tranquillity Chief')

    expect(after.mainTabCaption(STORED_ID, CANONICAL_TITLE)).toBe('Tranquillity Chief')
    expect(after.$botChatSessionIds.get().has(STORED_ID)).toBe(true)
  })

  it('keeps a side chat on its own stored title after the reload', async () => {
    const before = await launch()
    // A recent side chat opens with no tabTitle: its stored row names it.
    before.setSessionTileWorkspaceScope('side-1', { ...chiefScope, workspaceTabTitle: undefined })

    const after = await launch()
    after.setWorkspaceOwnerLabel(CHIEF_KEY, 'Tranquillity Chief')

    expect(after.mainTabCaption('side-1', 'Pricing notes')).toBe('Pricing notes')
    expect(after.$botChatSessionIds.get().has('side-1')).toBe(true)
  })

  it('stops captioning once the chat is re-opened as a working session', async () => {
    const before = await launch()
    before.setSessionTileWorkspaceScope(STORED_ID, chiefScope)
    before.setSessionTileWorkspaceScope(STORED_ID, { workspaceMode: 'sessions' })

    const after = await launch()
    after.setWorkspaceOwnerLabel(CHIEF_KEY, 'Tranquillity Chief')

    expect(after.mainTabCaption(STORED_ID, CANONICAL_TITLE)).toBe(CANONICAL_TITLE)
    expect(after.$botChatSessionIds.get().has(STORED_ID)).toBe(false)
  })
})

describe('upgrading from the stored list of bot-chat ids', () => {
  it('still marks the old ids as bot chats, and records their scope on the next open', async () => {
    window.localStorage.setItem('hermes.desktop.botChatSessions.v1', JSON.stringify([STORED_ID, 'other-bot-chat']))

    const upgraded = await launch()
    expect(upgraded.$botChatSessionIds.get().has(STORED_ID)).toBe(true)
    expect(upgraded.$botChatSessionIds.get().has('other-bot-chat')).toBe(true)
    // The old list never knew its owner: no caption until the chat is opened.
    expect(upgraded.$botChatScopes.get()[STORED_ID]).toBeUndefined()

    upgraded.setSessionTileWorkspaceScope(STORED_ID, chiefScope)

    const after = await launch()
    after.setWorkspaceOwnerLabel(CHIEF_KEY, 'Tranquillity Chief')
    expect(after.mainTabCaption(STORED_ID, CANONICAL_TITLE)).toBe('Tranquillity Chief')
    expect(after.$botChatSessionIds.get().has('other-bot-chat')).toBe(true)
    // One home for the family: the superseded key does not linger to be read back.
    expect(window.localStorage.getItem('hermes.desktop.botChatSessions.v1')).toBeNull()
  })
})

describe('the profile-keyed lifecycle', () => {
  it('drops a deleted profile’s remembered bot chat', async () => {
    const before = await launch()
    before.setSessionTileWorkspaceScope(STORED_ID, chiefScope)
    before.setSessionTileWorkspaceScope('alpha-chat', {
      ...chiefScope,
      ownerRoute: { connectionId: 'local', profile: 'alpha' },
      workspaceOwnerKey: 'bot:local::alpha'
    })

    before.dropTilesForProfile('tranquillity-chief')
    expect(before.$botChatScopes.get()[STORED_ID]).toBeUndefined()

    const after = await launch()
    expect(after.$botChatSessionIds.get().has(STORED_ID)).toBe(false)
    expect(after.$botChatScopes.get()['alpha-chat']?.workspaceOwnerKey).toBe('bot:local::alpha')
  })

  it('moves a renamed profile’s remembered bot chat route to the new name', async () => {
    const before = await launch()
    before.setSessionTileWorkspaceScope(STORED_ID, chiefScope)

    await before.migrateTilesForProfile('tranquillity-chief', 'tq-chief')
    expect(before.$botChatScopes.get()[STORED_ID]?.ownerRoute?.profile).toBe('tq-chief')

    const after = await launch()
    expect(after.$botChatScopes.get()[STORED_ID]?.ownerRoute).toEqual({
      connectionId: 'local',
      profile: 'tq-chief',
      targetProfile: 'tq-chief'
    })
    expect(after.$botChatSessionIds.get().has(STORED_ID)).toBe(true)
  })
})
