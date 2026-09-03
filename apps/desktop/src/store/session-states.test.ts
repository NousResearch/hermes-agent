import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { findGroupOfPane, group, split } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import {
  $workspaceMode,
  forgetActivePane,
  rememberActivePane,
  setWorkspaceScope,
  workspaceScopeKey
} from '@/components/pane-shell/workspace-scope'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $connection, $selectedStoredSessionId, setSessions } from '@/store/session'
import type { SessionProfileRoute } from '@/store/session-request-router'
import type { SessionTile } from '@/store/session-states'
import type * as SessionStatesModule from '@/store/session-states'
import {
  $focusedStoredSessionId,
  $sessionStates,
  $sessionTiles,
  blankDraftTile,
  clearAllSessionStates,
  closeAllOpenSessionTiles,
  focusedSessionNeedsRoute,
  focusOpenSession,
  focusWorkspaceOwnerSessionTile,
  foregroundSessionScopes,
  isSessionRemote,
  knownOwnerForSession,
  markSelectionRestore,
  nextSessionTileForWorkspace,
  openSessionTile,
  orderTilesByTree,
  patchSessionTile,
  recordSessionEventScope,
  releaseSessionTranscript,
  requestForOwnedSession,
  resetTileRuntimeBindings,
  selectionHomesToWorkspace,
  type SessionTileDelegate,
  sessionTileOwnerRoute,
  setSessionTileDelegate,
  setSessionTileWorkspaceScope
} from '@/store/session-states'

const tile = (storedSessionId: string): SessionTile => ({ storedSessionId })
const tilePane = (id: string) => `session-tile:${id}`

describe('foregroundSessionScopes', () => {
  beforeEach(() => {
    clearAllSessionStates()
    $activeSessionId.set(null)
    $sessionTiles.set([])
  })

  afterEach(() => {
    clearAllSessionStates()
    $activeSessionId.set(null)
    $sessionTiles.set([])
  })

  it('keeps the exact registry owner of an idle foreground runtime', () => {
    recordSessionEventScope({ connectionId: 'cloud', profile: 'default', session_id: 'runtime-1' })
    $activeSessionId.set('runtime-1')

    expect(foregroundSessionScopes()).toEqual(new Set(['conn:cloud::default']))
  })

  it('fails closed when the foreground runtime has no registered source', () => {
    $activeSessionId.set('legacy-runtime')

    expect(foregroundSessionScopes()).toEqual(new Set())
  })

  it('keeps every open pane owner, not only the focused runtime', () => {
    recordSessionEventScope({ connectionId: 'cloud-a', profile: 'default', session_id: 'runtime-a' })
    recordSessionEventScope({ connectionId: 'cloud-b', profile: 'default', session_id: 'runtime-b' })
    $sessionTiles.set([
      { runtimeId: 'runtime-a', storedSessionId: 'stored-a' },
      {
        ownerRoute: { connectionId: 'cloud-b', profile: 'default' },
        storedSessionId: 'stored-b'
      }
    ])

    expect(foregroundSessionScopes()).toEqual(new Set(['conn:cloud-a::default', 'conn:cloud-b::default']))
  })

  it('releases an idle pane owner when the pane closes', () => {
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: 'cloud', profile: 'default' },
        storedSessionId: 'stored-cloud'
      }
    ])

    expect(foregroundSessionScopes()).toEqual(new Set(['conn:cloud::default']))

    $sessionTiles.set([])

    expect(foregroundSessionScopes()).toEqual(new Set())
  })
})

describe('resetTileRuntimeBindings', () => {
  afterEach(() => {
    $sessionTiles.set([])
  })

  it('invalidates the delegate wiring cache AND drops tile runtime ids (sleep/wake reconnect)', () => {
    // The reconnect path must bust BOTH layers: the tile atoms' runtimeId and
    // the delegate's stored→runtime warm cache. Clearing only the atoms let
    // resumeTile's warm path re-bind the same dead runtime id after wake.
    const invalidateRuntimeBindings = vi.fn()
    setSessionTileDelegate({ invalidateRuntimeBindings } as unknown as SessionTileDelegate)

    $sessionTiles.set([{ runtimeId: 'runtime-dead', storedSessionId: 'stored-a' }])
    resetTileRuntimeBindings()

    expect(invalidateRuntimeBindings).toHaveBeenCalledTimes(1)
    expect($sessionTiles.get()).toEqual([
      { anchor: undefined, before: undefined, dir: undefined, storedSessionId: 'stored-a' }
    ])
  })

  it('tolerates a delegate without invalidateRuntimeBindings (older wiring)', () => {
    setSessionTileDelegate({} as unknown as SessionTileDelegate)
    $sessionTiles.set([{ runtimeId: 'runtime-dead', storedSessionId: 'stored-a' }])

    expect(() => resetTileRuntimeBindings()).not.toThrow()
    expect($sessionTiles.get()[0]?.runtimeId).toBeUndefined()
  })

  it('keeps Bot runtimes owned by a different connection', () => {
    const invalidateRuntimeBindings = vi.fn()
    setSessionTileDelegate({ invalidateRuntimeBindings } as unknown as SessionTileDelegate)
    $sessionTiles.set([
      {
        ownerRoute: {
          connectionId: 'barry',
          mode: 'remote',
          profile: 'oxcoder',
          targetProfile: 'oxcoder'
        },
        runtimeId: 'runtime-bot',
        storedSessionId: 'stored-bot',
        workspaceMode: 'bots',
        workspaceOwnerKey: 'bot:barry::oxcoder'
      }
    ])

    resetTileRuntimeBindings('work-vps')

    expect($sessionTiles.get()[0]?.runtimeId).toBe('runtime-bot')
    expect(invalidateRuntimeBindings).toHaveBeenCalledWith(new Set(['stored-bot']))
  })

  it('rebinds only the restarted connection while preserving other Bot gateways', () => {
    const invalidateRuntimeBindings = vi.fn()
    setSessionTileDelegate({ invalidateRuntimeBindings } as unknown as SessionTileDelegate)
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: 'barry', mode: 'remote', profile: 'oxcoder', targetProfile: 'oxcoder' },
        runtimeId: 'runtime-barry-dead',
        storedSessionId: 'stored-barry-bot',
        workspaceMode: 'bots',
        workspaceOwnerKey: 'bot:barry::oxcoder'
      },
      {
        ownerRoute: { connectionId: 'barry', mode: 'remote', profile: 't2oracle', targetProfile: 't2oracle' },
        runtimeId: 'runtime-barry-sibling-live',
        storedSessionId: 'stored-barry-sibling-bot',
        workspaceMode: 'bots',
        workspaceOwnerKey: 'bot:barry::t2oracle'
      },
      {
        ownerRoute: { connectionId: 'work-vps', mode: 'remote', profile: 'ceo', targetProfile: 'ceo' },
        runtimeId: 'runtime-work-live',
        storedSessionId: 'stored-work-bot',
        workspaceMode: 'bots',
        workspaceOwnerKey: 'bot:work-vps::ceo'
      },
      { runtimeId: 'runtime-session-dead', storedSessionId: 'stored-session' }
    ])

    resetTileRuntimeBindings({ connectionId: 'barry', profile: 'oxcoder' })

    const [barryBot, barrySibling, workBot, ordinarySession] = $sessionTiles.get()

    expect(barryBot).toMatchObject({ storedSessionId: 'stored-barry-bot' })
    expect(barryBot).not.toHaveProperty('runtimeId')
    expect(barrySibling).toMatchObject({
      runtimeId: 'runtime-barry-sibling-live',
      storedSessionId: 'stored-barry-sibling-bot'
    })
    expect(workBot).toMatchObject({ runtimeId: 'runtime-work-live', storedSessionId: 'stored-work-bot' })
    expect(ordinarySession).toMatchObject({ storedSessionId: 'stored-session' })
    expect(ordinarySession).not.toHaveProperty('runtimeId')
    expect(invalidateRuntimeBindings).toHaveBeenCalledWith(new Set(['stored-barry-sibling-bot', 'stored-work-bot']))
  })

  it('keeps an owner-routed SESSIONS tile (branch child) bound across an unrelated reconnect', () => {
    const invalidateRuntimeBindings = vi.fn()
    setSessionTileDelegate({ invalidateRuntimeBindings } as unknown as SessionTileDelegate)
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: '100-125-133-71-9119', mode: 'remote', profile: 'default' },
        runtimeId: 'runtime-branch-live',
        storedSessionId: 'stored-branch-child',
        workspaceMode: 'sessions'
      }
    ])

    // A flapping sibling connection reconnects; the branch child's runtime
    // lives on its parent's backend and must keep its binding — dropping it
    // re-arms the tile's resume, and repeated sibling flaps latch the
    // resume-storm error card over a healthy session.
    resetTileRuntimeBindings({ connectionId: 'other-ssh-source', profile: 'default' })

    expect($sessionTiles.get()[0]?.runtimeId).toBe('runtime-branch-live')
    expect(invalidateRuntimeBindings).toHaveBeenCalledWith(new Set(['stored-branch-child']))

    // Its OWN connection reconnecting still drops the binding for re-resume.
    resetTileRuntimeBindings({ connectionId: '100-125-133-71-9119', profile: 'default' })
    expect($sessionTiles.get()[0]?.runtimeId).toBeUndefined()
  })

  it('unknown restarted identity preserves only Bot runtimes owned by provably-live connections', () => {
    // Legacy remote primary: no registry connectionId to scope by. The dead
    // owner can't be named, so keep only owners we know are alive elsewhere —
    // the restarted backend's own Bot tile must still drop its binding.
    const invalidateRuntimeBindings = vi.fn()
    setSessionTileDelegate({ invalidateRuntimeBindings } as unknown as SessionTileDelegate)
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: 'legacy-remote', mode: 'remote', profile: 'writer', targetProfile: 'writer' },
        runtimeId: 'runtime-legacy-dead',
        storedSessionId: 'stored-legacy-bot',
        workspaceMode: 'bots',
        workspaceOwnerKey: 'bot:legacy-remote::writer'
      },
      {
        ownerRoute: { connectionId: 'work-vps', mode: 'remote', profile: 'ceo', targetProfile: 'ceo' },
        runtimeId: 'runtime-work-live',
        storedSessionId: 'stored-work-bot',
        workspaceMode: 'bots',
        workspaceOwnerKey: 'bot:work-vps::ceo'
      },
      { runtimeId: 'runtime-session-dead', storedSessionId: 'stored-session' }
    ])

    resetTileRuntimeBindings({ liveConnectionIds: new Set(['work-vps']) })

    const [legacyBot, workBot, ordinarySession] = $sessionTiles.get()

    expect(legacyBot).toMatchObject({ storedSessionId: 'stored-legacy-bot' })
    expect(legacyBot).not.toHaveProperty('runtimeId')
    expect(workBot).toMatchObject({ runtimeId: 'runtime-work-live', storedSessionId: 'stored-work-bot' })
    expect(ordinarySession).not.toHaveProperty('runtimeId')
    expect(invalidateRuntimeBindings).toHaveBeenCalledWith(new Set(['stored-work-bot']))
  })
})

describe('SessionTile workspace scope', () => {
  afterEach(() => {
    $activeGatewayProfile.set('default')
    $layoutTree.set(null)
    $selectedStoredSessionId.set(null)
    $sessionTiles.set([])
  })

  it('persists a sessions-mode owner route so a branch child tile pins its owning socket', () => {
    const ownerRoute = { connectionId: '100-125-133-71-9119', mode: 'remote' as const, profile: 'default' }

    openSessionTile('branch-child', 'center', undefined, null, { ownerRoute, workspaceMode: 'sessions' })

    expect($sessionTiles.get()).toEqual([
      expect.objectContaining({ ownerRoute, storedSessionId: 'branch-child', workspaceMode: 'sessions' })
    ])
  })

  it('keeps an existing sessions-mode owner route on a route-less re-scope', () => {
    const ownerRoute = { connectionId: '100-125-133-71-9119', mode: 'remote' as const, profile: 'default' }

    openSessionTile('branch-child', 'center', undefined, null, { ownerRoute, workspaceMode: 'sessions' })
    // A plain sidebar re-open routes through setSessionTileWorkspaceScope with
    // no route — absence of information, not a revocation.
    setSessionTileWorkspaceScope('branch-child', { workspaceMode: 'sessions' })

    expect($sessionTiles.get()).toEqual([expect.objectContaining({ ownerRoute, storedSessionId: 'branch-child' })])
  })

  it('stores an exact Bot owner and keeps it through placement patches', () => {
    const ownerRoute = {
      connectionId: 'connection-a',
      mode: 'remote' as const,
      profile: 'default',
      targetProfile: 'backend-default'
    }

    const scope = { ownerRoute, workspaceMode: 'bots' as const, workspaceOwnerKey: 'connection-a::default' }

    openSessionTile('bot-chat', 'right', undefined, undefined, scope)
    patchSessionTile('bot-chat', { dir: 'left' })

    expect($sessionTiles.get()).toEqual([
      expect.objectContaining({
        dir: 'left',
        ownerRoute,
        storedSessionId: 'bot-chat',
        workspaceMode: 'bots',
        workspaceOwnerKey: 'connection-a::default'
      })
    ])
  })

  it('allows a Bot-scoped tab when the same stored session is hidden in Sessions main', () => {
    const scope = { workspaceMode: 'bots' as const, workspaceOwnerKey: 'connection-a::default' }

    $selectedStoredSessionId.set('bot-chat')
    openSessionTile('bot-chat', 'center', undefined, undefined, scope)

    expect($sessionTiles.get()).toEqual([
      expect.objectContaining({
        storedSessionId: 'bot-chat',
        workspaceMode: 'bots',
        workspaceOwnerKey: 'connection-a::default'
      })
    ])
    expect(focusOpenSession('bot-chat', scope)).toBe('tile')
  })

  it('fronts the existing tab when compaction rotated the tip id — never a duplicate', () => {
    // The tile was opened when seg-2 was the tip; the conversation has since
    // rotated to seg-3 (projected row carries the full chain). Opening the
    // new tip must front that tile, not open the same chat twice.
    setSessions([{ _lineage_ids: ['seg-1', 'seg-2', 'seg-3'], _lineage_root_id: 'seg-1', id: 'seg-3' } as never])
    openSessionTile('seg-2')

    expect(focusOpenSession('seg-3')).toBe('tile')
    expect($sessionTiles.get().map(t => t.storedSessionId)).toEqual(['seg-2'])

    // The open path dedupes through the same lineage test.
    openSessionTile('seg-3')
    expect($sessionTiles.get().map(t => t.storedSessionId)).toEqual(['seg-2'])
    setSessions([])
  })

  it('keeps Bot tabs while a profile publication swaps the Sessions bucket', () => {
    const scope = { workspaceMode: 'bots' as const, workspaceOwnerKey: 'connection-a::writer' }

    openSessionTile('sessions-chat')
    openSessionTile('bot-chat', 'center', undefined, undefined, scope)
    $activeGatewayProfile.set('other-profile')

    expect($sessionTiles.get()).toEqual([
      expect.objectContaining({
        storedSessionId: 'bot-chat',
        workspaceMode: 'bots',
        workspaceOwnerKey: 'connection-a::writer'
      })
    ])
  })

  it('re-scopes an existing tile without changing its placement', () => {
    openSessionTile('chat', 'bottom', 'workspace')

    expect(
      setSessionTileWorkspaceScope('chat', {
        workspaceMode: 'bots',
        workspaceOwnerKey: 'connection-b::default'
      })
    ).toBe(true)
    expect($sessionTiles.get()[0]).toMatchObject({
      anchor: 'workspace',
      dir: 'bottom',
      workspaceMode: 'bots',
      workspaceOwnerKey: 'connection-b::default'
    })
  })

  it('preserves an existing Bot tile scope when moving it without an explicit scope', () => {
    const scope = {
      ownerRoute: {
        connectionId: 'connection-a',
        mode: 'remote' as const,
        profile: 'default',
        targetProfile: 'default'
      },
      workspaceMode: 'bots' as const,
      workspaceOwnerKey: 'bot:connection-a::default',
      workspaceTabTitle: 'Bot chat'
    }

    openSessionTile('bot-chat', 'right', undefined, undefined, scope)
    $layoutTree.set(group(['workspace', 'session-tile:bot-chat'], { id: 'workspace-group' }))
    // A split drag re-docks the tab with no scope (session-drag onCommit).
    openSessionTile('bot-chat', 'left', 'workspace')

    expect($sessionTiles.get()).toEqual([
      expect.objectContaining({
        anchor: 'workspace',
        dir: 'left',
        ownerRoute: scope.ownerRoute,
        storedSessionId: 'bot-chat',
        workspaceMode: 'bots',
        workspaceOwnerKey: scope.workspaceOwnerKey,
        workspaceTabTitle: 'Bot chat'
      })
    ])

    // An explicit scope from the caller still wins over the tile's current one.
    openSessionTile('bot-chat', 'right', 'workspace', undefined, { workspaceMode: 'sessions' })

    expect($sessionTiles.get()).toEqual([
      expect.objectContaining({ dir: 'right', storedSessionId: 'bot-chat', workspaceMode: 'sessions' })
    ])
    expect($sessionTiles.get()[0]).not.toHaveProperty('workspaceOwnerKey', scope.workspaceOwnerKey)
  })

  it('preserves workspace scope while dropping a stale runtime binding', () => {
    $sessionTiles.set([
      {
        runtimeId: 'runtime-dead',
        storedSessionId: 'bot-chat',
        workspaceMode: 'bots',
        workspaceOwnerKey: 'connection-a::default'
      }
    ])

    resetTileRuntimeBindings()

    expect($sessionTiles.get()[0]).toEqual({
      anchor: undefined,
      before: undefined,
      dir: undefined,
      storedSessionId: 'bot-chat',
      workspaceMode: 'bots',
      workspaceOwnerKey: 'connection-a::default'
    })
  })
})

describe('focusWorkspaceOwnerSessionTile', () => {
  const botA = { workspaceMode: 'bots' as const, workspaceOwnerKey: 'bot:a' }
  const botB = { workspaceMode: 'bots' as const, workspaceOwnerKey: 'bot:b' }

  afterEach(() => {
    forgetActivePane(workspaceScopeKey('bots', 'bot:a'))
    $layoutTree.set(null)
    $sessionTiles.set([])
  })

  it('reports null for an owner with no open tile — the caller opens something', () => {
    openSessionTile('other-bot-chat', 'center', 'workspace', undefined, botB)
    openSessionTile('sessions-chat')

    expect(focusWorkspaceOwnerSessionTile('bot:a')).toBeNull()
  })

  it('fronts the tab the owner last had active and reports its stored id', () => {
    openSessionTile('older-thread', 'center', 'workspace', undefined, botA)
    openSessionTile('newer-thread', 'center', 'workspace', undefined, botA)
    $layoutTree.set(
      group(['workspace', tilePane('older-thread'), tilePane('newer-thread')], { active: 'workspace', id: 'main' })
    )
    rememberActivePane(workspaceScopeKey('bots', 'bot:a'), tilePane('older-thread'))

    expect(focusWorkspaceOwnerSessionTile('bot:a')).toBe('older-thread')
    expect(findGroupOfPane($layoutTree.get()!, tilePane('older-thread'))?.active).toBe(tilePane('older-thread'))
  })

  it('falls back to the most recently opened tab when nothing is remembered', () => {
    openSessionTile('older-thread', 'center', 'workspace', undefined, botA)
    openSessionTile('newer-thread', 'center', 'workspace', undefined, botA)
    $layoutTree.set(
      group(['workspace', tilePane('older-thread'), tilePane('newer-thread')], { active: 'workspace', id: 'main' })
    )

    expect(focusWorkspaceOwnerSessionTile('bot:a')).toBe('newer-thread')
    expect(findGroupOfPane($layoutTree.get()!, tilePane('newer-thread'))?.active).toBe(tilePane('newer-thread'))
  })

  it('ignores a remembered tab that has since been closed', () => {
    openSessionTile('closed-bot-chat', 'center', 'workspace', undefined, botA)
    openSessionTile('thread', 'center', 'workspace', undefined, botA)
    rememberActivePane(workspaceScopeKey('bots', 'bot:a'), tilePane('closed-bot-chat'))
    $sessionTiles.set($sessionTiles.get().filter(t => t.storedSessionId !== 'closed-bot-chat'))

    expect(focusWorkspaceOwnerSessionTile('bot:a')).toBe('thread')
  })

  it("never crosses owners: another bot's open tabs do not count", () => {
    openSessionTile('other-bot-chat', 'center', 'workspace', undefined, botB)

    expect(focusWorkspaceOwnerSessionTile('bot:a')).toBeNull()
    expect($sessionTiles.get().map(t => t.storedSessionId)).toEqual(['other-bot-chat'])
  })

  describe('staleness probe (#90102): the tile bucket reconciles with backend truth before it wins', () => {
    it('discards a stale tile and reports null so the caller runs its authoritative open', () => {
      openSessionTile('stale-bot-chat', 'center', 'workspace', undefined, botA)

      expect(focusWorkspaceOwnerSessionTile('bot:a', tile => tile.storedSessionId === 'stale-bot-chat')).toBeNull()
      // Discard, not close: resurrecting the tile would just front the stale
      // session again on the next click.
      expect($sessionTiles.get()).toEqual([])
    })

    it('fronts the surviving fresh tile after discarding the stale one', () => {
      openSessionTile('stale-bot-chat', 'center', 'workspace', undefined, botA)
      openSessionTile('live-thread', 'center', 'workspace', undefined, botA)
      $layoutTree.set(
        group(['workspace', tilePane('stale-bot-chat'), tilePane('live-thread')], { active: 'workspace', id: 'main' })
      )
      // The stale tile is even the remembered one — the exact stuck shape.
      rememberActivePane(workspaceScopeKey('bots', 'bot:a'), tilePane('stale-bot-chat'))

      expect(focusWorkspaceOwnerSessionTile('bot:a', tile => tile.storedSessionId === 'stale-bot-chat')).toBe(
        'live-thread'
      )
      expect($sessionTiles.get().map(t => t.storedSessionId)).toEqual(['live-thread'])
    })

    it("only judges the probed owner's tiles — other owners keep theirs", () => {
      openSessionTile('other-bot-chat', 'center', 'workspace', undefined, botB)
      openSessionTile('stale-bot-chat', 'center', 'workspace', undefined, botA)

      expect(focusWorkspaceOwnerSessionTile('bot:a', () => true)).toBeNull()
      expect($sessionTiles.get().map(t => t.storedSessionId)).toEqual(['other-bot-chat'])
    })

    it('a throwing probe keeps the tile — reconciliation must not break the click', () => {
      openSessionTile('bot-chat', 'center', 'workspace', undefined, botA)

      expect(
        focusWorkspaceOwnerSessionTile('bot:a', () => {
          throw new Error('probe blew up')
        })
      ).toBe('bot-chat')
      expect($sessionTiles.get().map(t => t.storedSessionId)).toEqual(['bot-chat'])
    })

    it('no probe keeps the old behavior byte for byte', () => {
      openSessionTile('bot-chat', 'center', 'workspace', undefined, botA)

      expect(focusWorkspaceOwnerSessionTile('bot:a')).toBe('bot-chat')
      expect($sessionTiles.get().map(t => t.storedSessionId)).toEqual(['bot-chat'])
    })
  })
})

describe('closeAllOpenSessionTiles persists Bot Mode Close All (#94137)', () => {
  afterEach(() => {
    $activeGatewayProfile.set('default')
    $layoutTree.set(null)
    $sessionTiles.set([])
  })
  it('drops persisted bot tiles so a roster/profile rehydrate cannot restore them', () => {
    openSessionTile('chat-a', 'center', 'workspace', undefined, {
      workspaceMode: 'bots',
      workspaceOwnerKey: 'bot:a'
    })
    openSessionTile('chat-b', 'center', 'workspace', undefined, {
      workspaceMode: 'bots',
      workspaceOwnerKey: 'bot:b'
    })
    $layoutTree.set(group(['workspace', tilePane('chat-a'), tilePane('chat-b')], { active: 'workspace', id: 'main' }))
    closeAllOpenSessionTiles('workspace')
    expect($sessionTiles.get()).toEqual([])
    // Profile swap re-reads the shared Bot bucket. Close All must have
    // emptied it, not only dismissed the tree panes.
    $activeGatewayProfile.set('other-profile')
    expect($sessionTiles.get()).toEqual([])
    $activeGatewayProfile.set('default')
    expect($sessionTiles.get()).toEqual([])
  })
  it('leaves session tiles stacked in a different zone open', () => {
    openSessionTile('keep', 'right', 'workspace')
    openSessionTile('close-me', 'center', 'workspace')
    $layoutTree.set(
      split('row', [
        group(['workspace', tilePane('close-me')], { active: 'workspace', id: 'main' }),
        group([tilePane('keep')], { active: tilePane('keep'), id: 'right' })
      ])
    )
    closeAllOpenSessionTiles('workspace')
    expect($sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['keep'])
  })
})

describe('dropTilesForProfile', () => {
  const TILES_KEY = 'hermes.desktop.sessionTiles.v2'
  const BOTS_BUCKET = '__bots_workspace__'

  const storedTiles = (): Record<string, unknown> => {
    const raw = window.localStorage.getItem(TILES_KEY)

    return raw ? (JSON.parse(raw) as Record<string, unknown>) : {}
  }

  // The tile buckets are module-private and other suites leave persisted
  // entries behind, so each test re-imports a FRESH session-states module
  // (empty tilesByProfile + empty storage) instead of fighting the residue —
  // the same isolation the browser gets between app launches.
  type SessionStates = typeof SessionStatesModule
  let mod: SessionStates
  let activeGatewayProfile: { set: (name: string) => void }
  beforeEach(async () => {
    window.localStorage.clear()
    vi.resetModules()
    // Re-import the module graph so the private tile buckets start empty; the
    // profile atom session-states subscribes to is the freshly loaded one too.
    mod = await import('@/store/session-states')
    const profile = await import('@/store/profile')
    activeGatewayProfile = profile.$activeGatewayProfile
    activeGatewayProfile.set('default')
    mod.$sessionTiles.set([])
  })
  afterEach(() => {
    window.localStorage.clear()
    $activeGatewayProfile.set('default')
    $layoutTree.set(null)
    $selectedStoredSessionId.set(null)
    $sessionTiles.set([])
  })
  it("drops the deleted profile's persisted session tiles from memory and storage", () => {
    activeGatewayProfile.set('worker')
    mod.openSessionTile('worker-session-1')
    mod.openSessionTile('worker-session-2', 'left')
    expect(mod.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['worker-session-1', 'worker-session-2'])
    expect(storedTiles()).toHaveProperty('worker')
    mod.dropTilesForProfile('worker')
    expect(mod.$sessionTiles.get()).toEqual([])
    expect(storedTiles()).not.toHaveProperty('worker')
    // Session tiles of another profile survive the drop.
    activeGatewayProfile.set('writer')
    mod.openSessionTile('writer-session-1')
    expect(mod.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['writer-session-1'])
    expect(storedTiles()).toHaveProperty('writer')
  })
  it("drops Bot Mode tiles owned by a locally-deleted profile and keeps the other bots' tiles", () => {
    mod.openSessionTile('bot-chat-1', 'right', undefined, undefined, {
      ownerRoute: { connectionId: 'local', mode: 'local' as const, profile: 'researcher-1' },
      workspaceMode: 'bots' as const,
      workspaceOwnerKey: 'researcher-1'
    })
    mod.openSessionTile('bot-chat-2', 'right', undefined, undefined, {
      ownerRoute: { connectionId: 'local', mode: 'local' as const, profile: 'writer-1' },
      workspaceMode: 'bots' as const,
      workspaceOwnerKey: 'writer-1'
    })
    mod.dropTilesForProfile('researcher-1')
    expect(mod.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['bot-chat-2'])
    expect(
      (storedTiles()[BOTS_BUCKET] as Array<{ storedSessionId: string }>).map(tile => tile.storedSessionId)
    ).toEqual(['bot-chat-2'])
  })
  it('drops a source-scoped bot tile only for the exact route and keeps same-name agents elsewhere', () => {
    const route = {
      connectionId: 'source-a',
      mode: 'remote' as const,
      profile: 'worker',
      targetProfile: 'backend-worker'
    }

    mod.openSessionTile('bot-a', 'right', undefined, undefined, {
      ownerRoute: route,
      workspaceMode: 'bots' as const,
      workspaceOwnerKey: 'source-a::worker'
    })
    mod.openSessionTile('bot-b', 'right', undefined, undefined, {
      ownerRoute: {
        connectionId: 'source-b',
        mode: 'remote' as const,
        profile: 'worker',
        targetProfile: 'backend-worker'
      },
      workspaceMode: 'bots' as const,
      workspaceOwnerKey: 'source-b::worker'
    })
    mod.openSessionTile('bot-local', 'right', undefined, undefined, {
      ownerRoute: { connectionId: 'local', mode: 'local' as const, profile: 'worker' },
      workspaceMode: 'bots' as const,
      workspaceOwnerKey: 'local::worker'
    })
    mod.dropTilesForProfile('worker', route)
    expect(mod.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['bot-b', 'bot-local'])
    expect(
      (storedTiles()[BOTS_BUCKET] as Array<{ storedSessionId: string }>).map(tile => tile.storedSessionId)
    ).toEqual(['bot-b', 'bot-local'])
  })

  it('keeps a same-named bot tile owned by another connection when a local profile is deleted', () => {
    mod.openSessionTile('bot-local', 'right', undefined, undefined, {
      ownerRoute: { connectionId: 'local', mode: 'local' as const, profile: 'copilot' },
      workspaceMode: 'bots' as const,
      workspaceOwnerKey: 'copilot'
    })
    mod.openSessionTile('bot-remote', 'right', undefined, undefined, {
      ownerRoute: {
        connectionId: 'work-vps',
        mode: 'remote' as const,
        profile: 'copilot',
        targetProfile: 'copilot'
      },
      workspaceMode: 'bots' as const,
      workspaceOwnerKey: 'work-vps::copilot'
    })

    mod.dropTilesForProfile('copilot')

    // Only the LOCAL connection's tile points at the deleted local profile; the
    // same-named agent on another connection keeps its tile and conversation.
    expect(mod.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['bot-remote'])
    expect(
      (storedTiles()[BOTS_BUCKET] as Array<{ storedSessionId: string }>).map(tile => tile.storedSessionId)
    ).toEqual(['bot-remote'])
  })

  it('normalizes whitespace in the deleted identity on both delete paths', () => {
    mod.openSessionTile('bot-local', 'right', undefined, undefined, {
      ownerRoute: { connectionId: 'local', mode: 'local' as const, profile: 'press-bot' },
      workspaceMode: 'bots' as const,
      workspaceOwnerKey: 'press-bot'
    })
    mod.openSessionTile('bot-routed', 'right', undefined, undefined, {
      ownerRoute: {
        connectionId: 'source-a',
        mode: 'remote' as const,
        profile: 'worker',
        targetProfile: 'backend-worker'
      },
      workspaceMode: 'bots' as const,
      workspaceOwnerKey: 'source-a::worker'
    })

    // Non-route delete with a whitespace-padded name matches the trimmed tile.
    mod.dropTilesForProfile('   press-bot   ')
    expect(mod.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['bot-routed'])

    // Route delete with padded route fields matches the exact route's tile.
    mod.dropTilesForProfile('worker', {
      connectionId: ' source-a ',
      profile: '  worker  ',
      targetProfile: '  backend-worker  '
    })
    expect(mod.$sessionTiles.get()).toEqual([])
  })

  it('keeps profile identity case-exact on both delete paths', () => {
    mod.openSessionTile('bot-local', 'right', undefined, undefined, {
      ownerRoute: { connectionId: 'local', mode: 'local' as const, profile: 'press-bot' },
      workspaceMode: 'bots' as const,
      workspaceOwnerKey: 'press-bot'
    })
    mod.openSessionTile('bot-routed', 'right', undefined, undefined, {
      ownerRoute: {
        connectionId: 'source-a',
        mode: 'remote' as const,
        profile: 'worker',
        targetProfile: 'backend-worker'
      },
      workspaceMode: 'bots' as const,
      workspaceOwnerKey: 'source-a::worker'
    })

    // normalizeProfileKey trims but never lowercases: a differing case is a
    // different profile identity, consistently in the local and route branches.
    mod.dropTilesForProfile('PRESS-BOT')
    mod.dropTilesForProfile('worker', {
      connectionId: 'source-a',
      profile: 'WORKER',
      targetProfile: 'backend-worker'
    })

    expect(mod.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['bot-local', 'bot-routed'])
  })

  it('drops a legacy Bot tile whose ownerRoute predates the connectionId field', () => {
    // Tiles persisted before ownerRoute.connectionId existed (pre-#94235) carry
    // no connection id at all. `String(undefined ?? '').trim()` yields '', so a
    // local-delete branch comparing against `=== 'local'` never matches and the
    // tile survives every delete, resurrecting the deleted profile on relaunch
    // (hermes-agent#94235). The branch must treat a missing id as local.
    mod.openSessionTile('bot-legacy', 'right', undefined, undefined, {
      ownerRoute: { mode: 'local' as const, profile: 'press-bot' } as unknown as SessionProfileRoute,
      workspaceMode: 'bots' as const,
      workspaceOwnerKey: 'press-bot'
    })

    mod.dropTilesForProfile('press-bot')

    expect(mod.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual([])
    expect(storedTiles()).not.toHaveProperty(BOTS_BUCKET)
  })

  it("treats a missing connectionId as the canonical 'local' spelling", () => {
    // The local-delete branch compares ownerConnection to 'local', and
    // local-mode owner routes record that same spelling. No renderer-importable
    // constant exists (LOCAL_CONNECTION_ID lives in the electron main process),
    // so this pins the two spellings together: legacy (no id), canonical
    // ('local'), and divergent ('Local') tiles differ only in this field, and
    // only the first two may be dropped by a local delete. Renaming either side
    // fails the suite before it can silently orphan local Bot tiles
    // (Enough1122 review of #94426).
    mod.openSessionTile('bot-legacy', 'right', undefined, undefined, {
      ownerRoute: { mode: 'local' as const, profile: 'press-bot' } as unknown as SessionProfileRoute,
      workspaceMode: 'bots' as const,
      workspaceOwnerKey: 'press-bot'
    })
    mod.openSessionTile('bot-canonical', 'right', undefined, undefined, {
      ownerRoute: { connectionId: 'local', mode: 'local' as const, profile: 'press-bot' },
      workspaceMode: 'bots' as const,
      workspaceOwnerKey: 'press-bot'
    })
    mod.openSessionTile('bot-divergent', 'right', undefined, undefined, {
      ownerRoute: { connectionId: 'Local', mode: 'local' as const, profile: 'press-bot' },
      workspaceMode: 'bots' as const,
      workspaceOwnerKey: 'press-bot'
    })

    mod.dropTilesForProfile('press-bot')

    expect(mod.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['bot-divergent'])
  })

  it('throws on a route without profile instead of silently falling into the local-delete branch', () => {
    // A caller passing a route with only connectionId/targetProfile would
    // silently take the local branch and start requiring
    // `ownerConnection === 'local'` — dropping nothing remotely owned while
    // appearing to succeed. Both current call sites always populate profile, so
    // refuse the malformed shape loudly (Enough1122 review of #94426).
    mod.openSessionTile('bot-remote', 'right', undefined, undefined, {
      ownerRoute: {
        connectionId: 'work-vps',
        mode: 'remote' as const,
        profile: 'copilot',
        targetProfile: 'copilot'
      },
      workspaceMode: 'bots' as const,
      workspaceOwnerKey: 'work-vps::copilot'
    })

    expect(() => mod.dropTilesForProfile('copilot', { connectionId: 'work-vps', targetProfile: 'copilot' })).toThrow(
      /route without profile/
    )
    expect(mod.$sessionTiles.get().map(tile => tile.storedSessionId)).toEqual(['bot-remote'])
  })
})

describe('releaseSessionTranscript', () => {
  afterEach(() => {
    $sessionStates.set({})
  })

  it('normalizes legacy state whose messages field is undefined', () => {
    const legacy = { busy: false, storedSessionId: 'stored' } as ClientSessionState
    $sessionStates.set({ runtime: legacy })

    expect(() => releaseSessionTranscript('runtime')).not.toThrow()
    expect($sessionStates.get().runtime).toEqual({ ...legacy, messages: [] })
  })

  it('ignores a legacy undefined state without throwing', () => {
    $sessionStates.set({ runtime: undefined } as unknown as Record<string, ClientSessionState>)

    expect(() => releaseSessionTranscript('runtime')).not.toThrow()
    expect($sessionStates.get()).toHaveProperty('runtime', undefined)
  })
})

describe('orderTilesByTree', () => {
  it('no-ops (null) without a tree or below two tiles', () => {
    expect(orderTilesByTree(null, [tile('a'), tile('b')])).toBeNull()
    expect(orderTilesByTree(group([tilePane('a')]), [tile('a')])).toBeNull()
  })

  it('reorders tiles to layout-tree encounter order across a split', () => {
    const tree = split('row', [group(['workspace', tilePane('b')]), group([tilePane('a')])])

    expect(orderTilesByTree(tree, [tile('a'), tile('b')])).toEqual([tile('b'), tile('a')])
  })

  it('returns null when the array already matches strip order (skip persist)', () => {
    const tree = split('row', [group([tilePane('b')]), group([tilePane('a')])])

    expect(orderTilesByTree(tree, [tile('b'), tile('a')])).toBeNull()
  })

  it('sorts not-yet-adopted tiles after placed ones, stably', () => {
    const tree = group(['workspace', tilePane('b')])

    expect(orderTilesByTree(tree, [tile('a'), tile('b'), tile('c')])).toEqual([tile('b'), tile('a'), tile('c')])
  })
})

describe('selectionHomesToWorkspace', () => {
  const tiles = [tile('a'), tile('b')]

  it('homes for a null selection or a non-tile session', () => {
    expect(selectionHomesToWorkspace(null, tiles)).toBe(true)
    expect(selectionHomesToWorkspace('c', tiles)).toBe(true)
  })

  it('skips homing when the selected id is already an open tile', () => {
    expect(selectionHomesToWorkspace('a', tiles)).toBe(false)
  })
})

describe('nextSessionTileForWorkspace (⌘W promotion source)', () => {
  afterEach(() => {
    $layoutTree.set(null)
    $sessionTiles.set([])
  })

  it('prefers a tile stacked WITH the workspace tab (nearest-out)', () => {
    $layoutTree.set(group(['workspace', tilePane('a'), tilePane('b')], { active: 'workspace', id: 'main' }))
    $sessionTiles.set([tile('a'), tile('b')])

    expect(nextSessionTileForWorkspace()).toBe('a')
  })

  it('side-by-side layout: a tile in ANOTHER zone still promotes instead of dropping main to a fresh draft (#88924)', () => {
    // main zone holds only the workspace; the session tile lives in its own
    // zone beside it — db's three-pane report shape.
    $layoutTree.set(
      split('row', [
        group(['workspace'], { active: 'workspace', id: 'main' }),
        group([tilePane('side')], { active: tilePane('side'), id: 'right' })
      ])
    )
    $sessionTiles.set([tile('side')])

    expect(nextSessionTileForWorkspace()).toBe('side')
  })

  it('returns null when no live tile exists anywhere in the tree', () => {
    $layoutTree.set(
      split('row', [
        group(['workspace'], { active: 'workspace', id: 'main' }),
        group([tilePane('stale')], { active: tilePane('stale'), id: 'right' })
      ])
    )
    // Pane persisted in the tree but its tile is gone — must not promote a ghost.
    $sessionTiles.set([])

    expect(nextSessionTileForWorkspace()).toBe(null)
  })
})

describe('boot-restore selection homing (⌘R tab persistence)', () => {
  const mainGroup = () => group(['workspace', tilePane('t')], { active: tilePane('t'), id: 'main' })

  const activePane = () => {
    const tree = $layoutTree.get()

    return tree?.type === 'group' ? tree.active : null
  }

  it('a normal selection change fronts the workspace tab over an active tile', () => {
    $layoutTree.set(mainGroup())
    $selectedStoredSessionId.set('nav-1')

    expect(activePane()).toBe('workspace')
  })

  it('markSelectionRestore skips homing exactly once, so the persisted active tab survives a reload', () => {
    $layoutTree.set(mainGroup())
    markSelectionRestore()
    $selectedStoredSessionId.set('boot-1')

    // Boot restore: the tile tab the user reloaded on stays fronted.
    expect(activePane()).toBe(tilePane('t'))

    // One-shot consumed: the next selection change is a real navigation.
    $selectedStoredSessionId.set('nav-2')
    expect(activePane()).toBe('workspace')
  })
})

describe('$focusedStoredSessionId in Bot Mode (#96062)', () => {
  afterEach(() => {
    $layoutTree.set(null)
    $selectedStoredSessionId.set(null)
    setWorkspaceScope('sessions')
  })

  it('a Bots-pane click keeps the main-zone bot tile focused instead of collapsing to a null selection edge', () => {
    // Bot chats open as TILES and never set $selectedStoredSessionId. Clicking
    // a roster row moves the interaction tracker to the sidebar group, whose
    // active pane is chrome ('hermes-bots:pane'), not a session tile. The old
    // derivation then fell back to the null primary selection and published a
    // NULL "focused session" edge — which the Bots plugin read as "the chat
    // lost the center", releasing its open claim and re-asserting the Bots
    // home over the still-visible chat (the reported "jumps to the list").
    setWorkspaceScope('bots', 'bot:b')
    $selectedStoredSessionId.set(null)
    $layoutTree.set(
      split('row', [
        group(['sessions', 'hermes-bots:pane'], { active: 'hermes-bots:pane', id: 'grp-sessions' }),
        group(['workspace', tilePane('chat-b')], { active: tilePane('chat-b'), id: 'grp-main' })
      ])
    )
    noteActiveTreeGroup('grp-sessions')

    expect($focusedStoredSessionId.get()).toBe('chat-b')
  })

  it('the main-zone tile also answers while the tracker sits on the workspace tab itself', () => {
    setWorkspaceScope('bots', 'bot:b')
    $selectedStoredSessionId.set(null)
    $layoutTree.set(group(['workspace', tilePane('chat-b')], { active: tilePane('chat-b'), id: 'grp-main' }))
    noteActiveTreeGroup('grp-main')

    expect($focusedStoredSessionId.get()).toBe('chat-b')
  })

  it('a closed bot chat (no tile in main) still falls back to the selection', () => {
    setWorkspaceScope('bots', 'bot:b')
    $selectedStoredSessionId.set(null)
    $layoutTree.set(
      split('row', [
        group(['sessions', 'hermes-bots:pane'], { active: 'hermes-bots:pane', id: 'grp-sessions' }),
        group(['workspace'], { active: 'workspace', id: 'grp-main' })
      ])
    )
    noteActiveTreeGroup('grp-sessions')

    // No tile owns the main zone — the chat was closed — so the null edge is
    // genuine and must still surface (that is what lets the Bots home return).
    expect($focusedStoredSessionId.get()).toBeNull()
  })

  it('sessions mode keeps collapsing to the primary selection (derivation gated to Bot Mode)', () => {
    $selectedStoredSessionId.set('primary-1')
    $layoutTree.set(
      split('row', [
        group(['sessions'], { active: 'sessions', id: 'grp-sessions' }),
        group(['workspace', tilePane('stacked')], { active: tilePane('stacked'), id: 'grp-main' })
      ])
    )
    noteActiveTreeGroup('grp-sessions')

    // The main-zone tile must NOT answer here: in sessions mode the sidebar
    // highlight follows the primary selection exactly as it always has.
    expect($workspaceMode.get()).toBe('sessions')
    expect($focusedStoredSessionId.get()).toBe('primary-1')
  })
})

describe('focusedSessionNeedsRoute', () => {
  it('routes when the session is not on screen', () => {
    expect(focusedSessionNeedsRoute(null, false)).toBe(true)
    expect(focusedSessionNeedsRoute(null, true)).toBe(true)
  })

  it('routes for the ACTIVE main session while a full page covers the workspace', () => {
    expect(focusedSessionNeedsRoute('main', true)).toBe(true)
  })

  it('skips the route when the main session is already the visible chat', () => {
    expect(focusedSessionNeedsRoute('main', false)).toBe(false)
  })

  it('never routes for a tile — its pane shows the chat on any route', () => {
    expect(focusedSessionNeedsRoute('tile', true)).toBe(false)
    expect(focusedSessionNeedsRoute('tile', false)).toBe(false)
  })
})

describe('blankDraftTile', () => {
  const bound = (storedSessionId: string, runtimeId: string): SessionTile => ({ runtimeId, storedSessionId })

  const state = (messages: number, busy = false) =>
    ({ busy, messages: Array.from({ length: messages }, (_, i) => ({ id: `m${i}` })) }) as ClientSessionState

  it('finds the open tab whose session has no messages', () => {
    const tiles = [bound('a', 'run-a'), bound('b', 'run-b')]
    const states = { 'run-a': state(3), 'run-b': state(0) }

    expect(blankDraftTile(tiles, states)).toEqual(tiles[1])
  })

  it('picks the most recent blank tab when there are several', () => {
    const tiles = [bound('a', 'run-a'), bound('b', 'run-b')]
    const states = { 'run-a': state(0), 'run-b': state(0) }

    expect(blankDraftTile(tiles, states)).toEqual(tiles[1])
  })

  it('leaves a blank-but-busy tab alone — its first turn is already in flight', () => {
    expect(blankDraftTile([bound('a', 'run-a')], { 'run-a': state(0, true) })).toBeNull()
  })

  it('treats an unbound or unpublished tile as unknown, not empty', () => {
    expect(blankDraftTile([tile('a')], {})).toBeNull()
    expect(blankDraftTile([bound('a', 'run-a')], {})).toBeNull()
  })

  it('is null when every open tab holds a conversation', () => {
    expect(blankDraftTile([bound('a', 'run-a')], { 'run-a': state(2) })).toBeNull()
    expect(blankDraftTile([], {})).toBeNull()
  })
})

// ⌘⇧T used to only restore `$sessionTiles`. Adoption inserts silently
// (activate:false), so the tab came back behind the still-fronted workspace.
// Real path: register, adopt, focus — same as paneMirror + reopen.
describe('reopenLastClosedTile focuses the restored tab', () => {
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

    // panes ← $sessionTiles (paneMirror stub). Adoption is synchronous on
    // register, so openSessionTile + focusOpenSession works the same tick.
    const registered = new Map<string, () => void>()

    const syncTiles = () => {
      const wanted = new Set(states.$sessionTiles.get().map(t => t.storedSessionId))

      for (const id of wanted) {
        if (registered.has(id)) {
          continue
        }

        registered.set(
          id,
          registry.register({
            area: 'panes',
            data: { dock: { pane: 'workspace', pos: 'center' }, placement: 'main' },
            id: tilePane(id),
            render: () => null,
            title: id
          })
        )
      }

      for (const [id, dispose] of registered) {
        if (!wanted.has(id)) {
          dispose()
          registered.delete(id)
          tree.removeTreePane(tilePane(id))
        }
      }
    }

    states.$sessionTiles.listen(syncTiles)
    tree.watchContributedPanes()
    session.$selectedStoredSessionId.set('primary')
    tree.declareDefaultTree(model.group(['workspace'], { active: 'workspace', id: 'grp-main' }))

    states.openSessionTile('closed', 'center', 'workspace')
    states.focusOpenSession('closed')
    tree.noteActiveTreeGroup('grp-main')
    expect(findGroupOfPane(tree.$layoutTree.get()!, tilePane('closed'))?.active).toBe(tilePane('closed'))

    return { states, tree }
  }

  it('fronts the restored tab after ⌘⇧T', async () => {
    const { states, tree } = await setup()

    states.closeSessionTile('closed')
    expect(states.$sessionTiles.get().some(t => t.storedSessionId === 'closed')).toBe(false)
    expect(findGroupOfPane(tree.$layoutTree.get()!, 'workspace')?.active).toBe('workspace')

    states.reopenLastClosedTile()

    expect(states.$sessionTiles.get().some(t => t.storedSessionId === 'closed')).toBe(true)
    expect(findGroupOfPane(tree.$layoutTree.get()!, tilePane('closed'))?.active).toBe(tilePane('closed'))
    expect(tree.$activeTreeGroup.get()).toBe('grp-main')
  })
})

describe('sessionTileOwnerRoute', () => {
  afterEach(() => {
    $sessionTiles.set([])
  })

  it('returns the exact owning route a bot chat tile was opened with', () => {
    // This is what lets a bot chat RPC reach the bot's OWN local gateway even
    // while chrome stays on the launch profile: the tile carries the route, so
    // the request router never has to guess from the (hidden, unlisted) row.
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: 'local', mode: 'local', profile: 'developer' },
        storedSessionId: 'bot-chat-developer'
      }
    ])

    expect(sessionTileOwnerRoute('bot-chat-developer')).toEqual({
      connectionId: 'local',
      mode: 'local',
      profile: 'developer'
    })
  })

  it('returns undefined for a tile with no owner route (plain session)', () => {
    $sessionTiles.set([{ storedSessionId: 'plain' }])

    expect(sessionTileOwnerRoute('plain')).toBeUndefined()
  })

  it('returns undefined when the session has no tile', () => {
    $sessionTiles.set([])

    expect(sessionTileOwnerRoute('missing')).toBeUndefined()
  })
})

describe('knownOwnerForSession / requestForOwnedSession (#91684 client half)', () => {
  beforeEach(() => {
    // Earlier suites leave profile-scoped tile buckets behind; pin the profile
    // BEFORE seeding tiles so the bucket subscriber cannot clobber the seed.
    $activeGatewayProfile.set('default')
    $sessionTiles.set([])
  })
  afterEach(() => {
    $sessionTiles.set([])
    setSessions([])
  })

  it('resolves the tile owner route first, translating a runtime id to its stored id', () => {
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: 'conn-a', profile: 'work' },
        runtimeId: 'rt-1',
        storedSessionId: 'stored-1'
      }
    ])

    expect(knownOwnerForSession('rt-1')).toEqual({ connectionId: 'conn-a', profile: 'work' })
    expect(knownOwnerForSession('stored-1')).toEqual({ connectionId: 'conn-a', profile: 'work' })
  })

  it('falls back to the session row profile when no tile route exists', () => {
    setSessions([{ id: 'stored-2', profile: 'loki' } as never])

    expect(knownOwnerForSession('stored-2')).toBe('loki')
  })

  it('keeps a session row connection owner when profiles share the same name', () => {
    setSessions([{ connection_id: 'source-b', id: 'stored-shared', profile: 'default' } as never])

    expect(knownOwnerForSession('stored-shared')).toEqual({ connectionId: 'source-b', profile: 'default' })
  })

  it('returns undefined (ambient) when no owner is known, and for null ids', () => {
    expect(knownOwnerForSession('unknown-session')).toBeUndefined()
    expect(knownOwnerForSession(null)).toBeUndefined()
    expect(knownOwnerForSession(undefined)).toBeUndefined()
  })

  it('requestForOwnedSession dispatches ambient with the exact 2-arg shape when no owner is known', async () => {
    const ambient = vi.fn(async (method: string, params?: Record<string, unknown>) => ({ method, params }))

    await requestForOwnedSession('unknown-session', ambient as never, 'approval.respond', {
      choice: 'once',
      session_id: 'unknown-session'
    })

    expect(ambient).toHaveBeenCalledTimes(1)
    expect(ambient.mock.calls[0]).toEqual(['approval.respond', { choice: 'once', session_id: 'unknown-session' }])
  })
})

describe('isSessionRemote (#94640)', () => {
  beforeEach(() => {
    $activeGatewayProfile.set('default')
    $sessionTiles.set([])
  })
  afterEach(() => {
    $sessionTiles.set([])
    setSessions([])
    $connection.set(null)
  })

  it('falls back to the ambient connection when the session has no known owner route', () => {
    $connection.set({ mode: 'remote' } as never)
    expect(isSessionRemote('unknown-session')).toBe(true)

    $connection.set({ mode: 'local' } as never)
    expect(isSessionRemote('unknown-session')).toBe(false)
  })

  it("prefers the session's OWN owner route over an ambient connection of a different mode", () => {
    // The window's active/ambient connection is local, but this session
    // belongs to a registered secondary REMOTE connection (Bot Mode / the
    // unified Sessions list). Composer image uploads must still upload
    // bytes for it — reading ambient mode here shipped a client-local
    // composer-images path to the remote backend (#94640).
    $connection.set({ mode: 'local' } as never)
    $sessionTiles.set([
      {
        ownerRoute: { connectionId: 'homelab', mode: 'remote', profile: 'default' },
        runtimeId: 'rt-1',
        storedSessionId: 'stored-1'
      }
    ])

    expect(isSessionRemote('rt-1')).toBe(true)
    expect(isSessionRemote('stored-1')).toBe(true)
  })

  it('falls back to ambient when the owner is a bare pool profile (no connectionId/mode)', () => {
    $connection.set({ mode: 'remote' } as never)
    setSessions([{ id: 'stored-2', profile: 'loki' } as never])

    expect(isSessionRemote('stored-2')).toBe(true)
  })
})
