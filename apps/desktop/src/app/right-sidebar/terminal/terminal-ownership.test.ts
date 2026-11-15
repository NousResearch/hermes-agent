import { waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// Real terminal, connection and session stores; only the native terminal IPC
// is stubbed. Every test loads fresh modules so a "relaunch" re-reads storage.
async function loadStores() {
  vi.resetModules()
  const terminals = await import('./terminals')
  const gateway = await import('./terminal-gateway')
  const attempt = await import('./terminal-session-attempt')
  const { $connectionsRegistry } = await import('@/store/connections')
  const { $connection, $currentCwd } = await import('@/store/session')

  return { ...terminals, ...gateway, ...attempt, $connection, $connectionsRegistry, $currentCwd }
}

type Stores = Awaited<ReturnType<typeof loadStores>>

const registryWith = (...ssh: Array<{ host: string; id: string }>) =>
  ({
    version: 2,
    primary: 'local',
    connections: [
      { id: 'local', kind: 'local', label: 'This device' },
      ...ssh.map(entry => ({ ...entry, kind: 'ssh', label: entry.id.toUpperCase() }))
    ]
  }) as never

function controllerFor(stores: Stores, id: string, start: ReturnType<typeof vi.fn>) {
  const write = vi.fn()

  const controller = stores.createSessionAttemptController(id, {
    isDisposed: () => false,
    isReplayingHistory: () => false,
    onStarted: () => {},
    resolveCwd: () => '/repo',
    restoreHistory: () => {},
    sessionIdRef: { current: null },
    setStatus: () => {},
    term: { clearSelection: () => {}, cols: 80, rows: 24, write } as never,
    terminalApi: {
      attach: vi.fn(async () => true),
      dispose: vi.fn(async () => true),
      onData: () => () => {},
      onExit: () => () => {},
      start,
      write: vi.fn()
    } as never,
    writeOutput: () => {}
  })

  return { controller, write }
}

const session = (connectionId: string) => ({ connectionId, cwd: '/repo', id: `pty-${connectionId}`, shell: 'bash' })

beforeEach(() => {
  window.localStorage.clear()
  delete window.document.documentElement.dataset.hermesDesktopHost
})

describe('a terminal tab owns the gateway it was opened on', () => {
  it('a failed first start retries on the same gateway after a Sessions pick', async () => {
    const stores = await loadStores()
    stores.$connectionsRegistry.set(registryWith({ host: 'nas.example', id: 'nas' }))
    stores.$connection.set({ connectionId: 'nas' } as never)

    // The plain + button: no explicit gateway.
    const id = stores.createTerminal('/repo')
    expect(stores.$terminals.get().find(term => term.id === id)).toMatchObject({ connectionId: 'nas' })
    const start = vi.fn().mockRejectedValueOnce(new Error('NAS offline')).mockResolvedValue(session('nas'))
    const { controller, write } = controllerFor(stores, id, start)

    controller.start()
    await waitFor(() => expect(write).toHaveBeenCalled())
    expect(start.mock.calls[0]![0].connectionId).toBe('nas')

    stores.$connection.set({ connectionId: 'local' } as never)
    controller.input('\r')
    await waitFor(() => expect(start).toHaveBeenCalledTimes(2))
    controller.release()

    expect(start.mock.calls[1]![0].connectionId).toBe('nas')
    expect(stores.$terminals.get().find(term => term.id === id)?.connectionId).toBe('nas')
  })

  it('a restored tab with no gateway pins the one it first tried, even when that start fails', async () => {
    const stores = await loadStores()
    const id = stores.createTerminal('/repo') // no registry yet: nothing to record

    stores.$connectionsRegistry.set(registryWith({ host: 'nas.example', id: 'nas' }))
    stores.$connection.set({ connectionId: 'nas' } as never)
    const start = vi.fn().mockRejectedValueOnce(new Error('NAS offline')).mockResolvedValue(session('nas'))
    const { controller, write } = controllerFor(stores, id, start)

    controller.start()
    await waitFor(() => expect(write).toHaveBeenCalled())
    stores.$connection.set({ connectionId: 'local' } as never)
    controller.input('\r')
    await waitFor(() => expect(start).toHaveBeenCalledTimes(2))
    controller.release()

    expect(start.mock.calls[1]![0].connectionId).toBe('nas')
  })

  it('the same literal cwd on another host selects that host’s tab', async () => {
    const stores = await loadStores()
    stores.$connectionsRegistry.set(registryWith({ host: 'nas.example', id: 'nas' }))
    stores.$connection.set({ connectionId: 'nas' } as never)
    stores.$currentCwd.set('/elsewhere')

    const nas = stores.createTerminal('/repo', { connectionId: 'nas', profile: 'default' })
    const local = stores.createTerminal('/repo', { connectionId: 'local' })
    stores.selectTerminal(nas)
    stores.$currentCwd.set('/repo')
    expect(stores.$activeTerminalId.get()).toBe(nas)

    stores.$connection.set({ connectionId: 'local' } as never)
    stores.$currentCwd.set('/repo')
    expect(stores.$activeTerminalId.get()).toBe(local)

    stores.$connection.set({ connectionId: 'nas' } as never)
    expect(stores.$activeTerminalId.get()).toBe(nas)

    // A manual pick survives while neither host nor cwd changes.
    stores.selectTerminal(local)
    stores.$connectionsRegistry.set(registryWith({ host: 'nas.example', id: 'nas' }))
    expect(stores.$activeTerminalId.get()).toBe(local)
  })
})

describe('a removed gateway stays removed for its tabs', () => {
  it('deleting NAS and adding another machine named NAS never reconnects the old tab, across relaunch', async () => {
    let stores = await loadStores()
    stores.$connectionsRegistry.set(registryWith({ host: 'old-nas.example', id: 'nas' }))
    const id = stores.createTerminal('/repo', { connectionId: 'nas', profile: 'default' })

    stores.$connectionsRegistry.set(registryWith())
    stores.$connectionsRegistry.set(registryWith({ host: 'different-prod.example', id: 'nas' }))
    const tab = () => stores.$terminals.get().find(term => term.id === id)!
    expect(stores.terminalHost(tab(), stores.$connectionsRegistry.get())).toBe('removed')

    stores = await loadStores() // relaunch
    stores.$connectionsRegistry.set(registryWith({ host: 'different-prod.example', id: 'nas' }))
    expect(stores.terminalHost(tab(), stores.$connectionsRegistry.get())).toBe('removed')

    const start = vi.fn().mockResolvedValue(session('nas'))
    const { controller, write } = controllerFor(stores, id, start)
    controller.start()
    await waitFor(() => expect(write).toHaveBeenCalled())
    controller.release()

    expect(start).not.toHaveBeenCalled()
  })

  it('editing NAS in place keeps its tabs reconnecting to it, across relaunch', async () => {
    let stores = await loadStores()
    stores.$connectionsRegistry.set(registryWith({ host: 'old-nas.example', id: 'nas' }))
    const id = stores.createTerminal('/repo', { connectionId: 'nas', profile: 'default' })
    stores.$connectionsRegistry.set(registryWith({ host: 'new-nas.example', id: 'nas' }))

    stores = await loadStores() // relaunch
    stores.$connectionsRegistry.set(registryWith({ host: 'new-nas.example', id: 'nas' }))
    const tab = stores.$terminals.get().find(term => term.id === id)!
    expect(stores.terminalHost(tab, stores.$connectionsRegistry.get())).toMatchObject({ connection: { id: 'nas' } })

    const start = vi.fn().mockResolvedValue(session('nas'))
    const { controller } = controllerFor(stores, id, start)
    controller.start()
    await waitFor(() => expect(start).toHaveBeenCalledTimes(1))
    controller.release()

    expect(start.mock.calls[0]![0]).toMatchObject({ connectionId: 'nas', profile: 'default' })
  })
})
