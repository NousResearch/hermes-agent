import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $workspaceMode,
  $workspaceNewSessionTarget,
  $workspaceOwnerKey,
  forgetActivePane,
  forgetRememberedPane,
  rememberActivePane,
  resetRememberedActivePanes,
  resolveRememberedActivePane,
  setWorkspaceOwnerLabel,
  setWorkspaceScope,
  workspaceOwnerTitle,
  workspaceSessionBaseTitle,
  workspaceSessionRenameable
} from './workspace-scope'

afterEach(() => {
  setWorkspaceScope('sessions')
})

describe('workspace scope', () => {
  it('publishes a coherent mode and owner in one batch', () => {
    const snapshots: Array<['sessions' | 'bots', string | null]> = []
    const capture = () => snapshots.push([$workspaceMode.get(), $workspaceOwnerKey.get()])
    const unbindMode = $workspaceMode.listen(capture)
    const unbindOwner = $workspaceOwnerKey.listen(capture)
    snapshots.length = 0

    expect(setWorkspaceScope('bots', 'connection-a::default')).toBe(true)
    expect(snapshots.length).toBeGreaterThan(0)
    expect(snapshots.every(snapshot => snapshot[0] === 'bots' && snapshot[1] === 'connection-a::default')).toBe(true)
    expect(setWorkspaceScope('bots', 'connection-a::default')).toBe(false)

    unbindMode()
    unbindOwner()
  })

  it('publishes the exact new-session route with its Bot owner', () => {
    const route = {
      connectionId: 'connection-a',
      mode: 'remote' as const,
      profile: 'writer',
      targetProfile: 'writer'
    }

    expect(setWorkspaceScope('bots', 'bot:connection-a::writer', { kind: 'route', route })).toBe(true)
    expect($workspaceNewSessionTarget.get()).toEqual({ kind: 'route', route })

    // Equivalent route objects are a semantic no-op, not a new render signal.
    expect(setWorkspaceScope('bots', 'bot:connection-a::writer', { kind: 'route', route: { ...route } })).toBe(false)

    setWorkspaceScope('sessions')
    expect($workspaceNewSessionTarget.get()).toBeNull()
  })
})

describe('workspace owner title', () => {
  it('captions a bot chat by its bot instead of the canonical stored title, and leaves everything else alone (#99152)', () => {
    setWorkspaceOwnerLabel('bot:alpha', 'Alpha')
    const botChat = { workspaceMode: 'bots' as const, workspaceOwnerKey: 'bot:alpha', workspaceTabTitle: 'Bot Chat' }

    expect(workspaceOwnerTitle('Bot Chat', botChat)).toBe('Alpha')
    // A `+` side thread under the same bot keeps its own title.
    expect(workspaceOwnerTitle('Plan the launch', botChat)).toBe('Plan the launch')
    // A Sessions tab titled the same way is not a bot chat.
    expect(workspaceOwnerTitle('Bot Chat', { workspaceMode: 'sessions' })).toBe('Bot Chat')
    // No label yet (roster not loaded): the stored title stands.
    expect(workspaceOwnerTitle('Bot Chat', { ...botChat, workspaceOwnerKey: 'bot:beta' })).toBe('Bot Chat')
  })
})

describe('workspace session base title', () => {
  it('resolves a hidden bot chat to its canonical title so the caption can name the bot (#99152 follow-up)', () => {
    setWorkspaceOwnerLabel('bot:alpha', 'Alpha')
    const botChat = { workspaceMode: 'bots' as const, workspaceOwnerKey: 'bot:alpha', workspaceTabTitle: 'Bot Chat' }

    // A canonical Bot Chat is hidden: it is in no sidebar slice, so the row
    // lookup returns nothing even while the chat is on screen.
    const base = workspaceSessionBaseTitle(undefined, botChat, 'New session')

    expect(base).toBe('Bot Chat')
    // The composed caption is what the main tab registers — the bug was that
    // this path fell straight through to "New session".
    expect(workspaceOwnerTitle(base, botChat)).toBe('Alpha')
  })

  it('lets a loaded row title win over the scope, and treats a blank one as absent', () => {
    const botChat = { workspaceMode: 'bots' as const, workspaceOwnerKey: 'bot:alpha', workspaceTabTitle: 'Bot Chat' }

    expect(workspaceSessionBaseTitle('Verify hermes profile session', botChat, 'New session')).toBe(
      'Verify hermes profile session'
    )
    expect(workspaceSessionBaseTitle('   ', botChat, 'New session')).toBe('Bot Chat')
    expect(workspaceSessionBaseTitle(null, botChat, 'New session')).toBe('Bot Chat')
  })

  it('keeps the placeholder for a session tab and for a bot side thread', () => {
    // A Sessions tab has no canonical bot title to inherit.
    expect(workspaceSessionBaseTitle(undefined, { workspaceMode: 'sessions' }, 'New session')).toBe('New session')
    expect(workspaceSessionBaseTitle(undefined, undefined, 'New session')).toBe('New session')
    // A `+` side thread under the same bot is not the canonical chat.
    expect(
      workspaceSessionBaseTitle(undefined, { workspaceMode: 'bots', workspaceOwnerKey: 'bot:alpha' }, 'New session')
    ).toBe('New session')
  })
})

describe('workspace session renameable', () => {
  // Same discriminator as the caption: only the bot workspace's OWN canonical
  // tab (workspaceMode 'bots' + a registered canonical tab title) loses the
  // verb. A session that merely happens to be open in Bot Mode keeps it, and
  // plain Sessions tabs are untouched (#124857).
  it('gates Rename on the canonical bot-chat scope exactly where the caption takes over', () => {
    const botChat = { workspaceMode: 'bots' as const, workspaceOwnerKey: 'bot:alpha', workspaceTabTitle: 'Bot Chat' }

    expect(workspaceSessionRenameable(botChat)).toBe(false)
    // A `+` side thread under the same bot has no canonical tab title.
    expect(workspaceSessionRenameable({ workspaceMode: 'bots' })).toBe(true)
    // A Sessions tab titled the same way is not a bot chat.
    expect(workspaceSessionRenameable({ workspaceMode: 'sessions', workspaceTabTitle: 'Bot Chat' })).toBe(true)
    // No scope at all (a plain row/tile): renameable.
    expect(workspaceSessionRenameable(undefined)).toBe(true)
    expect(workspaceSessionRenameable({})).toBe(true)
  })
})

describe('remembered active panes', () => {
  beforeEach(() => resetRememberedActivePanes())

  it('remembers and restores panes independently per owner key', () => {
    rememberActivePane('conn-a:profile-x', 'pane-1')
    rememberActivePane('conn-b:profile-y', 'pane-2')

    expect(resolveRememberedActivePane('conn-a:profile-x', ['pane-1', 'pane-2'])).toBe('pane-1')
    expect(resolveRememberedActivePane('conn-b:profile-y', ['pane-1', 'pane-2'])).toBe('pane-2')
  })

  it('does not collide on a shared profile suffix across owner keys', () => {
    rememberActivePane('local:main', 'pane-local')

    expect(resolveRememberedActivePane('ssh:server:main', [])).toBeNull()
  })

  it('falls back after the remembered pane is removed', () => {
    rememberActivePane('bot-a', 'pane-gone')

    expect(resolveRememberedActivePane('bot-a', ['first', 'second'])).toBe('first')
    expect(resolveRememberedActivePane('bot-a', [])).toBeNull()
  })

  it('forgets a single owner without touching others', () => {
    rememberActivePane('bot-a', 'pane-a')
    rememberActivePane('bot-b', 'pane-b')

    forgetActivePane('bot-a')

    expect(resolveRememberedActivePane('bot-a', ['fallback-a', 'pane-a'])).toBe('fallback-a')
    expect(resolveRememberedActivePane('bot-b', ['pane-a', 'pane-b'])).toBe('pane-b')
  })

  it('forgets a removed pane across every owner that remembered it', () => {
    rememberActivePane('bot-a', 'pane-gone')
    rememberActivePane('bot-b', 'pane-gone')

    forgetRememberedPane('pane-gone')

    expect(resolveRememberedActivePane('bot-a', ['fallback-a'])).toBe('fallback-a')
    expect(resolveRememberedActivePane('bot-b', ['fallback-b'])).toBe('fallback-b')
  })
})
