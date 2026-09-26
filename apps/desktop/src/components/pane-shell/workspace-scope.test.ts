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
  workspaceMainSessionRenamable,
  workspaceMainSessionScope,
  workspaceOwnerTitle,
  workspaceSessionRenamable,
  workspaceSessionTitle,
  workspaceSessionUsesDraftTitle
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

  it('uses the Bot Chat identity when its hidden row is absent from the visible session list', () => {
    setWorkspaceOwnerLabel('bot:hermes', 'Hermes')

    const botChat = {
      workspaceMode: 'bots' as const,
      workspaceOwnerKey: 'bot:hermes',
      workspaceTabTitle: 'Bot Chat'
    }

    expect(workspaceSessionTitle(null, 'New session', botChat)).toBe('Hermes')
    expect(workspaceSessionTitle(null, 'New session', undefined)).toBe('New session')
    expect(workspaceSessionUsesDraftTitle(false, botChat)).toBe(false)
    expect(workspaceSessionUsesDraftTitle(false, undefined)).toBe(true)
    expect(workspaceSessionUsesDraftTitle(true, undefined)).toBe(false)
    expect(workspaceSessionRenamable(botChat)).toBe(false)
    expect(workspaceSessionRenamable(undefined)).toBe(true)
  })

  it('falls back to the active Bot owner when a hidden lineage tip misses the exact-id scope cache', () => {
    setWorkspaceOwnerLabel('bot:wallstreetscout', 'Wallstreetscout')

    const scope = workspaceMainSessionScope(undefined, 'bots', 'bot:wallstreetscout')

    expect(scope).toEqual({
      workspaceMode: 'bots',
      workspaceOwnerKey: 'bot:wallstreetscout',
      workspaceTabTitle: 'Bot Chat'
    })
    expect(workspaceSessionTitle(null, 'New session', scope)).toBe('Wallstreetscout')
    expect(workspaceSessionUsesDraftTitle(false, scope)).toBe(false)
    expect(workspaceMainSessionRenamable('bots', false, undefined)).toBe(false)
  })

  it('keeps an ordinary Sessions main draft titled and renamable as before', () => {
    const scope = workspaceMainSessionScope(undefined, 'sessions', null)

    expect(scope).toBeUndefined()
    expect(workspaceSessionTitle(null, 'New session', scope)).toBe('New session')
    expect(workspaceSessionUsesDraftTitle(false, scope)).toBe(true)
    expect(workspaceMainSessionRenamable('sessions', false, scope)).toBe(true)
    expect(workspaceMainSessionRenamable('bots', true, scope)).toBe(true)
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
