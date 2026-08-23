import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $workspaceMode,
  $workspaceNewSessionTarget,
  $workspaceOwnerKey,
  contributesToWorkspace,
  filterContributionsForWorkspace,
  forgetActivePane,
  forgetRememberedPane,
  rememberActivePane,
  resetRememberedActivePanes,
  resolveRememberedActivePane,
  setWorkspaceScope
} from './workspace-scope'

interface ScopedContribution {
  id: string
  workspaceMode?: 'sessions' | 'bots'
  workspaceOwnerKey?: string
}

const contribution = (
  id: string,
  scope?: Pick<ScopedContribution, 'workspaceMode' | 'workspaceOwnerKey'>
): ScopedContribution => ({ id, ...scope })

const bot = (ownerKey: string, suffix: string) =>
  contribution(`bot:${suffix}`, {
    workspaceMode: 'bots',
    workspaceOwnerKey: ownerKey
  })

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

  it('keeps a group owner explicit while blocking generic session creation', () => {
    const target = { kind: 'blocked' as const, message: 'New group conversations start in the group composer.' }

    setWorkspaceScope('bots', 'group:room-1', target)

    expect($workspaceOwnerKey.get()).toBe('group:room-1')
    expect($workspaceNewSessionTarget.get()).toEqual(target)
  })

  it('keeps global contributions visible in both modes', () => {
    expect(contributesToWorkspace(undefined, 'sessions', null)).toBe(true)
    expect(contributesToWorkspace(undefined, 'bots', 'bot-a')).toBe(true)
  })

  it('separates sessions and bots contributions by mode', () => {
    const sessionsOnly = contribution('sessions-pane', { workspaceMode: 'sessions' })
    const botsOnly = bot('bot-a', 'pane')

    expect(contributesToWorkspace(sessionsOnly, 'sessions')).toBe(true)
    expect(contributesToWorkspace(sessionsOnly, 'bots', 'bot-a')).toBe(false)
    expect(contributesToWorkspace(botsOnly, 'sessions')).toBe(false)
    expect(contributesToWorkspace(botsOnly, 'bots', 'bot-a')).toBe(true)
  })

  it('requires an exact non-empty owner match for bots contributions', () => {
    const scoped = bot('bot-a', 'pane')

    expect(contributesToWorkspace(scoped, 'bots', 'bot-b')).toBe(false)
    expect(contributesToWorkspace(scoped, 'bots', null)).toBe(false)
    expect(contributesToWorkspace(scoped, 'bots', '')).toBe(false)
    expect(contributesToWorkspace(scoped, 'bots', 'bot-a')).toBe(true)

    // A bots-scoped contribution with an empty owner key never participates.
    const noOwner = contribution('no-owner', { workspaceMode: 'bots' })
    expect(contributesToWorkspace(noOwner, 'bots', 'bot-a')).toBe(false)
  })

  it('does not collide on shared profile suffixes across connection-qualified keys', () => {
    // Same profile suffix, different connections — opaque exact strings only.
    const localProfile = bot('local:main', 'main')
    const remoteProfile = bot('ssh:server:main', 'main')

    expect(contributesToWorkspace(localProfile, 'bots', 'ssh:server:main')).toBe(false)
    expect(contributesToWorkspace(remoteProfile, 'bots', 'local:main')).toBe(false)
    expect(contributesToWorkspace(localProfile, 'bots', 'local:main')).toBe(true)
    expect(contributesToWorkspace(remoteProfile, 'bots', 'ssh:server:main')).toBe(true)
  })
})

describe('filterContributionsForWorkspace', () => {
  it('filters to the current mode and preserves input order', () => {
    const contributions = [
      bot('bot-a', 'zeta'),
      contribution('global-1'),
      contribution('sessions-only', { workspaceMode: 'sessions' }),
      bot('bot-a', 'alpha')
    ]

    expect(filterContributionsForWorkspace(contributions, 'bots', 'bot-a').map(c => c.id)).toEqual([
      'bot:zeta',
      'global-1',
      'bot:alpha'
    ])
    expect(filterContributionsForWorkspace(contributions, 'sessions', null).map(c => c.id)).toEqual([
      'global-1',
      'sessions-only'
    ])
  })

  it('returns the original array reference on a no-op', () => {
    const contributions: ScopedContribution[] = [contribution('a'), contribution('b')]

    expect(filterContributionsForWorkspace(contributions, 'sessions', null)).toBe(contributions)
    expect(filterContributionsForWorkspace(contributions, 'bots', 'anything').length).toBe(2)
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
