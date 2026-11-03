/**
 * A bot's tab reads as its BOT, not as the stored "Bot Chat", right after a
 * relaunch. Two inputs feed that caption and both used to be window-local:
 * - the owner label (bot -> display name), published only while the Bots pane
 *   renders, so every restored bot tab read "Bot Chat" until it was opened;
 * - the main tab's bot-chat scope, so a Bot Chat restored into main read
 *   "Bot Chat" until it was opened from its bot again.
 * A relaunch is a fresh module graph over the same localStorage.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const botChat = { workspaceMode: 'bots' as const, workspaceOwnerKey: 'bot:alpha', workspaceTabTitle: 'Bot Chat' }

async function launch() {
  vi.resetModules()

  return {
    scope: await import('./bot-chat-scope'),
    workspace: await import('@/components/pane-shell/workspace-scope')
  }
}

/** The main tab's caption for a stored row, as controller.tsx composes it. */
function mainCaption(app: Awaited<ReturnType<typeof launch>>, storedSessionId: string, rowTitle: string) {
  const scope = app.scope.$botChatScopes.get()[storedSessionId]

  return app.workspace.workspaceOwnerTitle(app.workspace.workspaceStoredTitle(rowTitle, scope) ?? rowTitle, scope)
}

beforeEach(() => window.localStorage.clear())
afterEach(() => window.localStorage.clear())

describe('a bot tab keeps its bot name across a relaunch', () => {
  it('captions a restored main-tab Bot Chat by its bot before the Bots pane publishes anything', async () => {
    const before = await launch()
    before.workspace.setWorkspaceOwnerLabel('bot:alpha', 'Alpha')
    before.scope.rememberBotChatScope('chat-1', botChat)
    expect(mainCaption(before, 'chat-1', 'Bot Chat')).toBe('Alpha')

    const after = await launch()

    expect(after.workspace.$workspaceOwnerLabels.get()).toEqual({ 'bot:alpha': 'Alpha' })
    expect(after.scope.$botChatScopes.get()['chat-1']).toEqual(botChat)
    expect(mainCaption(after, 'chat-1', 'Bot Chat')).toBe('Alpha')
  })

  it('keeps the scope a compression rotation carried onto the tip', async () => {
    const before = await launch()
    before.workspace.setWorkspaceOwnerLabel('bot:alpha', 'Alpha')
    before.scope.rememberBotChatScope('segment-1', botChat)
    before.scope.carryBotChatScope('segment-1', 'segment-2')

    const after = await launch()

    expect(mainCaption(after, 'segment-2', 'Bot Chat')).toBe('Alpha')
  })

  it('follows a bot rename, and forgets a chat reopened as a plain session', async () => {
    const before = await launch()
    before.workspace.setWorkspaceOwnerLabel('bot:alpha', 'Alpha')
    before.workspace.setWorkspaceOwnerLabel('bot:alpha', 'Alpha Prime')
    before.scope.rememberBotChatScope('chat-1', botChat)
    before.scope.rememberBotChatScope('chat-1', { workspaceMode: 'sessions' })

    const after = await launch()

    expect(after.workspace.$workspaceOwnerLabels.get()['bot:alpha']).toBe('Alpha Prime')
    expect(after.scope.$botChatScopes.get()['chat-1']).toBeUndefined()
    expect(mainCaption(after, 'chat-1', 'Bot Chat')).toBe('Bot Chat')
  })

  it('ignores malformed persisted values instead of captioning from them', async () => {
    window.localStorage.setItem('hermes.desktop.workspaceOwnerLabels.v1', JSON.stringify({ 'bot:alpha': 7 }))
    window.localStorage.setItem(
      'hermes.desktop.botChatScopes.v1',
      JSON.stringify({ 'chat-1': { workspaceMode: 'sessions' }, 'chat-2': 'bots' })
    )

    const app = await launch()

    expect(app.workspace.$workspaceOwnerLabels.get()).toEqual({})
    expect(app.scope.$botChatScopes.get()).toEqual({})
  })
})
