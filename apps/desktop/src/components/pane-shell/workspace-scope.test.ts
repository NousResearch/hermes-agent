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
  workspaceOwnerTitle
} from './workspace-scope'

afterEach(() => {
  setWorkspaceScope('sessions')
})

describe('workspace scope', () => {
  it('defaults to the un-switched sessions window state', () => {
    expect($workspaceMode.get()).toBe('sessions')
    expect($workspaceOwnerKey.get()).toBeNull()
    expect($workspaceNewSessionTarget.get()).toBeNull()
  })

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

  it('keeps a group owner explicit while publishing why it has no generic route', () => {
    const target = { kind: 'blocked' as const, message: 'New group conversations start in the group composer.' }

    setWorkspaceScope('bots', 'group:room-1', target)

    expect($workspaceOwnerKey.get()).toBe('group:room-1')
    expect($workspaceNewSessionTarget.get()).toEqual(target)
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
