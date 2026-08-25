import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { findGroupOfPane, group, split } from '@/components/pane-shell/tree/model'
import { $layoutTree } from '@/components/pane-shell/tree/store'
import { $activeGatewayProfile } from '@/store/profile'
import { $selectedStoredSessionId, setSessions } from '@/store/session'
import type { SessionTile } from '@/store/session-states'
import {
  $sessionStates,
  $sessionTiles,
  blankDraftTile,
  focusedSessionNeedsRoute,
  focusOpenSession,
  knownOwnerForSession,
  markSelectionRestore,
  nextSessionTileForWorkspace,
  openSessionTile,
  orderTilesByTree,
  patchSessionTile,
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
