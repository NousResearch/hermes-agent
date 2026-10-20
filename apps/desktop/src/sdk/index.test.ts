import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ackComposerInsert } from '@/app/chat/composer/focus'
import { createClientSessionState } from '@/lib/chat-runtime'
import { host } from '@/sdk'
import { $selectedStoredSessionId, setActiveSessionId, setAwaitingResponse, setBusy } from '@/store/session'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'

// The warm path must route through the guarded prewarm resolver (hover dwell,
// per-profile throttle), not dial the gateway directly: mocking the store's
// dialers lets the tests observe the ONLY side effect that matters — whether
// openGatewayForProfile was dialed.
const warmMocks = vi.hoisted(() => ({
  openGatewayForAgent: vi.fn(async (_connectionId: null | string, _profile: string) => undefined),
  openGatewayForProfile: vi.fn(async (_profile: string) => undefined)
}))

vi.mock('@/store/gateway', async importOriginal => ({
  ...((await importOriginal()) as Record<string, unknown>),
  openGatewayForAgent: warmMocks.openGatewayForAgent,
  openGatewayForProfile: warmMocks.openGatewayForProfile
}))

describe('host.warmProfile routing contract', () => {
  beforeEach(() => {
    warmMocks.openGatewayForProfile.mockClear()
    warmMocks.openGatewayForAgent.mockClear()
  })

  it('dials the profile through the guarded prewarm path', () => {
    host.warmProfile('warm-free-slot')

    expect(warmMocks.openGatewayForProfile).toHaveBeenCalledWith('warm-free-slot')
  })

  it('ignores an empty profile name', () => {
    host.warmProfile('   ')

    expect(warmMocks.openGatewayForProfile).not.toHaveBeenCalled()
  })
})

describe('host.state turn flags', () => {
  afterEach(() => {
    setActiveSessionId(null)
    setBusy(false)
    setAwaitingResponse(false)
    clearAllSessionStates()
  })

  it('uses the draft atoms when there is no runtime session', () => {
    expect(host.state.busy.get()).toBe(false)
    expect(host.state.awaitingResponse.get()).toBe(false)

    setBusy(true)
    setAwaitingResponse(true)

    expect(host.state.busy.get()).toBe(true)
    expect(host.state.awaitingResponse.get()).toBe(true)
  })

  it('reads the focused session slice once a runtime exists', () => {
    setBusy(false)
    setAwaitingResponse(false)
    setActiveSessionId('rt-focus')
    publishSessionState('rt-focus', {
      ...createClientSessionState('stored-focus'),
      awaitingResponse: true,
      busy: true
    })

    expect(host.state.busy.get()).toBe(true)
    expect(host.state.awaitingResponse.get()).toBe(true)

    publishSessionState('rt-focus', {
      ...createClientSessionState('stored-focus'),
      awaitingResponse: false,
      busy: true
    })

    expect(host.state.busy.get()).toBe(true)
    expect(host.state.awaitingResponse.get()).toBe(false)
  })

  it('does not pick up a background session', () => {
    setActiveSessionId('rt-focus')
    publishSessionState('rt-focus', createClientSessionState('stored-focus'))
    publishSessionState('rt-bg', {
      ...createClientSessionState('stored-bg'),
      awaitingResponse: true,
      busy: true
    })

    expect(host.state.busy.get()).toBe(false)
    expect(host.state.awaitingResponse.get()).toBe(false)
  })

  it('follows a focused session tile, not the primary', async () => {
    const tree = await import('@/components/pane-shell/tree/store')
    const model = await import('@/components/pane-shell/tree/model')
    const { registry } = await import('@/contrib/registry')
    const { $sessionTiles } = await import('@/store/session-states')

    // A second chat zone holding a session tile, next to the main workspace.
    for (const id of ['workspace', 'session-tile:tile-a']) {
      registry.register({
        area: 'panes',
        data: id === 'workspace' ? { placement: 'main', uncloseable: true } : { placement: 'main' },
        id,
        render: () => null,
        title: id
      })
    }

    tree.declareDefaultTree(
      model.split('row', [
        model.group(['workspace'], { active: 'workspace', id: 'grp-main' }),
        model.group(['session-tile:tile-a'], { active: 'session-tile:tile-a', id: 'grp-side' })
      ])
    )

    // Primary chat is idle; the tile's session is mid-turn.
    setActiveSessionId('rt-primary')
    publishSessionState('rt-primary', createClientSessionState('stored-primary'))
    $sessionTiles.set([{ runtimeId: 'rt-tile-a', storedSessionId: 'tile-a' }])
    publishSessionState('rt-tile-a', {
      ...createClientSessionState('tile-a'),
      awaitingResponse: true,
      busy: true
    })

    // Focusing the tile zone moves the flags onto the tile's session…
    tree.noteActiveTreeGroup('grp-side')
    expect(host.state.busy.get()).toBe(true)
    expect(host.state.awaitingResponse.get()).toBe(true)

    // …and homing back to the workspace returns to the (idle) primary.
    tree.noteActiveTreeGroup('grp-main')
    expect(host.state.busy.get()).toBe(false)
    expect(host.state.awaitingResponse.get()).toBe(false)

    $sessionTiles.set([])
  })
})

describe('host.connections', () => {
  const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
  const originalDesktop = desktopWindow.hermesDesktop

  const connection = (id: string, label: string) => ({
    id,
    kind: 'remote' as const,
    label,
    tokenPreview: null,
    tokenSet: true,
    url: `https://${id}.example`
  })

  const stubBridge = (list: () => Promise<unknown>) => {
    desktopWindow.hermesDesktop = {
      ...originalDesktop,
      connections: { list }
    } as unknown as Window['hermesDesktop']
  }

  afterEach(() => {
    desktopWindow.hermesDesktop = originalDesktop
  })

  it('returns the registry rows, not the envelope that carries them (#89823)', async () => {
    stubBridge(async () => ({
      connections: [connection('local', 'This Mac'), connection('homelab', 'Homelab')],
      primary: 'local',
      secureTokenStorage: true,
      version: 2
    }))

    const connections = await host.connections()

    expect(Array.isArray(connections)).toBe(true)
    expect(connections.map(entry => entry.id)).toEqual(['local', 'homelab'])
    expect(connections[1]).toMatchObject({ kind: 'remote', label: 'Homelab', url: 'https://homelab.example' })
  })

  it('folds the envelope-level primary id down onto the row that owns it', async () => {
    stubBridge(async () => ({
      connections: [connection('local', 'This Mac'), connection('homelab', 'Homelab')],
      primary: 'homelab',
      secureTokenStorage: true,
      version: 2
    }))

    expect((await host.connections()).map(entry => [entry.id, entry.primary])).toEqual([
      ['local', false],
      ['homelab', true]
    ])
  })

  it('reads as a single-source desktop when the payload carries no rows', async () => {
    stubBridge(async () => ({ primary: '', secureTokenStorage: true, version: 1 }))

    await expect(host.connections()).resolves.toEqual([])
  })

  it('still rejects on a Desktop build without the connection registry', async () => {
    desktopWindow.hermesDesktop = undefined

    await expect(host.connections()).rejects.toThrow('This Desktop build has no connection registry')
  })
})

describe('host workspace scope', () => {
  afterEach(async () => {
    host.setWorkspaceScope('sessions')
    const tree = await import('@/components/pane-shell/tree/store')
    tree.$newSessionTabAction.set(null)
    tree.removeTreePane('plugin-workspace:scope-test')
  })

  it('uses the shared tab action for an exact Bot owner without moving Sessions', async () => {
    const tree = await import('@/components/pane-shell/tree/store')
    const { $workspaceNewSessionTarget } = await import('@/components/pane-shell/workspace-scope')
    const opened: string[] = []

    const route = {
      connectionId: 'connection-b',
      mode: 'remote' as const,
      profile: 'writer',
      targetProfile: 'writer'
    }

    tree.$newSessionTabAction.set(() => opened.push('tab'))
    host.newChat(route, { workspaceMode: 'bots', workspaceOwnerKey: 'bot:connection-b::writer' })

    expect(opened).toEqual(['tab'])
    expect($workspaceNewSessionTarget.get()).toEqual({ kind: 'route', route })
  })
})

describe('host.composer draft facade', () => {
  afterEach(() => {
    setActiveSessionId(null)
    $selectedStoredSessionId.set(null)
    clearAllSessionStates()
  })

  it('routes insertText and focus by address: tile for a session, resolved-active for null', async () => {
    const seen: string[] = []

    const onInsert = (event: Event) => {
      const { mode, target, token } = (event as CustomEvent).detail

      seen.push(`insert:${mode}:${target}`)
      // A mounted surface acknowledges the insert like the real composer does.
      ackComposerInsert(token, true)
    }

    const onFocus = (event: Event) => seen.push(`focus:${(event as CustomEvent<{ target: string }>).detail.target}`)

    window.addEventListener('hermes:composer-insert', onInsert)
    window.addEventListener('hermes:composer-focus', onFocus)

    const [tileOk, activeOk] = await Promise.all([
      host.composer.insertText('sess-1', ' snippet ', { mode: 'inline' }),
      host.composer.insertText(null, 'to active')
    ])

    host.composer.focus('sess-1')
    host.composer.focus(null)
    // requestComposerFocus defers a plain focus request one macrotask.
    await new Promise(resolve => window.setTimeout(resolve, 0))

    window.removeEventListener('hermes:composer-insert', onInsert)
    window.removeEventListener('hermes:composer-focus', onFocus)

    expect([tileOk, activeOk]).toEqual([true, true])
    // A session id never resolves to the primary unless the primary shows it —
    // an absent tile drops the request rather than reaching the wrong pane.
    expect(seen).toEqual(['insert:inline:tile:sess-1', 'insert:block:main', 'focus:tile:sess-1', 'focus:main'])
  })

  it("addresses 'new' to the session-less primary composer only, never to the active one", async () => {
    const seen: string[] = []

    const onInsert = (event: Event) => {
      const { target, token } = (event as CustomEvent).detail

      seen.push(`insert:${target}`)
      ackComposerInsert(token, true)
    }

    const onFocus = (event: Event) => seen.push(`focus:${(event as CustomEvent<{ target: string }>).detail.target}`)

    window.addEventListener('hermes:composer-insert', onInsert)
    window.addEventListener('hermes:composer-focus', onFocus)

    // The primary shows a session → nothing hosts the new draft; the verbs
    // fail closed instead of landing in whatever composer the bus routes to.
    setActiveSessionId('rt-1')
    $selectedStoredSessionId.set('sess-1')
    await expect(host.composer.insertText('new', 'x')).resolves.toBe(false)
    expect(host.composer.submit('new', 'x')).toBe(false)
    host.composer.focus('new')
    await new Promise(resolve => window.setTimeout(resolve, 5))
    expect(seen).toEqual([])

    // No session in the primary → it IS the new draft.
    setActiveSessionId(null)
    $selectedStoredSessionId.set(null)
    await expect(host.composer.insertText('new', 'x')).resolves.toBe(true)
    host.composer.focus('new')
    await new Promise(resolve => window.setTimeout(resolve, 5))

    window.removeEventListener('hermes:composer-insert', onInsert)
    window.removeEventListener('hermes:composer-focus', onFocus)

    expect(seen).toEqual(['insert:main', 'focus:main'])
  })

  it('falls back to the persisted stash, keyed by the stored id, when no surface answers', async () => {
    const { stashSessionDraft } = await import('@/store/composer')

    stashSessionDraft('sess-stash', 'stashed draft', [])
    // The stash is keyed by the durable id; a plugin holding the runtime id
    // must still reach it once the session states map runtime → stored.
    publishSessionState('rt-stash', createClientSessionState('stored-stash'))
    stashSessionDraft('stored-stash', 'runtime-addressed', [])

    await expect(host.composer.getDraft('sess-stash')).resolves.toBe('stashed draft')
    await expect(host.composer.getDraft('rt-stash')).resolves.toBe('runtime-addressed')
  })
})

describe('host.sessions session-list mutations', () => {
  beforeEach(async () => {
    const layout = await import('@/store/layout')
    const color = await import('@/store/session-color')
    const session = await import('@/store/session')

    layout.$pinnedSessionIds.set([])
    layout.$sidebarSessionOrderIds.set([])
    layout.$sidebarSessionOrderManual.set(false)
    color.$sessionColorOverrides.set({})
    session.$sessions.set([])
  })

  it('pin/unpin write the pinned store the sidebar reads', async () => {
    const { $pinnedSessionIds } = await import('@/store/layout')

    host.sessions.pin('row-1')
    expect($pinnedSessionIds.get()).toEqual(['row-1'])

    host.sessions.pin('row-2')
    expect($pinnedSessionIds.get()).toEqual(['row-1', 'row-2'])

    host.sessions.pin('row-1', false)
    expect($pinnedSessionIds.get()).toEqual(['row-2'])

    // Drop-target pinning (drag-to-pin) slots the pin at an index instead of
    // appending — the same `pinSession(id, index)` the sidebar's drop uses.
    host.sessions.pin('row-0', true, 0)
    expect($pinnedSessionIds.get()).toEqual(['row-0', 'row-2'])
  })

  it('resolves a live id to its durable lineage root before pinning', async () => {
    const { $pinnedSessionIds } = await import('@/store/layout')
    const { $sessions } = await import('@/store/session')
    const { makeSessionInfo } = await import('@/test/session-info')

    $sessions.set([makeSessionInfo({ _lineage_root_id: 'root-9', id: 'tip-9' })])

    host.sessions.pin('tip-9')

    expect($pinnedSessionIds.get()).toEqual(['root-9'])
  })

  it('reorder persists the manual order the drag path writes, in the LIVE id space', async () => {
    const { $sidebarSessionOrderIds, $sidebarSessionOrderManual } = await import('@/store/layout')
    const { $sessions } = await import('@/store/session')
    const { makeSessionInfo } = await import('@/test/session-info')

    // `c` was compressed: the row slot hands a plugin its durable root `c`,
    // but the order store (and the sidebar's reconcile effect) key rows by
    // the live id `c-tip`. Feeding the durable id back verbatim would drop
    // the row from the order and flip the manual flag off on the next render.
    $sessions.set([makeSessionInfo({ _lineage_root_id: 'c', id: 'c-tip' }), makeSessionInfo({ id: 'a' })])

    host.sessions.reorder(['c', 'a', 'b'])

    expect($sidebarSessionOrderManual.get()).toBe(true)
    expect($sidebarSessionOrderIds.get()).toEqual(['c-tip', 'a', 'b'])

    // Empty list = clear the manual order, back to the default sort.
    host.sessions.reorder([])

    expect($sidebarSessionOrderManual.get()).toBe(false)
    expect($sidebarSessionOrderIds.get()).toEqual([])
  })

  it('reorderPinned permutes the Pinned section through the same setter the drag uses', async () => {
    const { $pinnedSessionIds } = await import('@/store/layout')
    const { $sessions } = await import('@/store/session')
    const { makeSessionInfo } = await import('@/test/session-info')

    $sessions.set([makeSessionInfo({ _lineage_root_id: 'p1', id: 'p1-tip' })])
    $pinnedSessionIds.set(['p1', 'p2', 'unloaded'])

    // Durable ids (the slot's) and live ids both resolve; an unmentioned pin keeps its slot.
    host.sessions.reorderPinned(['p2', 'p1-tip'])

    expect($pinnedSessionIds.get()).toEqual(['p2', 'p1', 'unloaded'])
  })

  it('setColor writes the durable-keyed colour override and clears with null', async () => {
    const { $sessionColorOverrides } = await import('@/store/session-color')

    host.sessions.setColor('row-1', '#ff8800')
    expect($sessionColorOverrides.get()).toEqual({ 'row-1': '#ff8800' })

    host.sessions.setColor('row-1', null)
    expect($sessionColorOverrides.get()).toEqual({})
  })
})
