import { afterEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { host } from '@/sdk'
import { setActiveSessionId, setAwaitingResponse, setBusy } from '@/store/session'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'

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

  it('registers plugin workspace ownership and chrome options', async () => {
    const { registry } = await import('@/contrib/registry')

    const close = host.openWorkspace('scope-test', {
      dock: { pane: 'workspace', pos: 'right' },
      headerVeto: true,
      render: () => null,
      title: 'Scoped',
      uncloseable: true,
      workspaceMode: 'bots',
      workspaceOwnerKey: 'connection-a::default'
    })

    expect(registry.getArea('panes').find(pane => pane.id === 'plugin-workspace:scope-test')).toMatchObject({
      data: {
        dock: { pane: 'workspace', pos: 'right' },
        headerVeto: true,
        uncloseable: true
      },
      workspaceMode: 'bots',
      workspaceOwnerKey: 'connection-a::default'
    })

    close()
  })

  it('publishes the active workspace scope through one host seam', async () => {
    const { $workspaceMode, $workspaceOwnerKey } = await import('@/components/pane-shell/workspace-scope')

    expect(host.setWorkspaceScope('bots', 'connection-b::default')).toBe(true)
    expect($workspaceMode.get()).toBe('bots')
    expect($workspaceOwnerKey.get()).toBe('connection-b::default')
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
