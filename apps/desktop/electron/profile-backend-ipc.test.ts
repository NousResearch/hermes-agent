import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import type { IpcMain } from 'electron'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  type ConnectionRegistry,
  normalizeConnectionInput,
  normalizeRegistry,
  upsertConnection
} from './connection-registry'
import {
  createGatewayConnectionsStore,
  GATEWAY_CONNECTIONS_FILE,
  type GatewayConnectionsStore
} from './gateway-connections-store'
import { type ProfileBackendIpcDeps, registerProfileBackendIpc } from './profile-backend-ipc'

// The user answers every native confirm with its first button (Allow).
vi.mock('electron', () => ({
  BrowserWindow: { fromWebContents: () => null },
  dialog: { showMessageBox: vi.fn(async () => ({ response: 0 })) }
}))

const LAUNCHER = { plugin: 'openshell', profile: 'work' }

const dirs: string[] = []

afterEach(() => {
  for (const dir of dirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

// The machine's saved-gateways list as main and the Webapp each open it: one file and one lock, separate caches.
function savedGateways() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-profile-backend-'))
  dirs.push(dir)

  const open = () =>
    createGatewayConnectionsStore({
      root: path.join(dir, 'hermes'),
      userData: path.join(dir, 'userData'),
      legacyRegistry: () => normalizeRegistry(null),
      decrypt: secret => String((secret as { value?: string } | null)?.value || '')
    })

  return { file: path.join(dir, 'hermes', GATEWAY_CONNECTIONS_FILE), main: open(), webapp: open() }
}

function saveGateway(store: GatewayConnectionsStore, label: string, url: string) {
  const registry = store.read()
  const entry = normalizeConnectionInput({ kind: 'remote', label, url }, registry)

  store.write(upsertConnection(registry, entry))
}

const labels = (registry: ConnectionRegistry) => registry.connections.map(c => c.label)

// The four handlers over an in-memory connection.json, with the deps main.ts hands them.
function profileBackendIpc(overrides: Partial<ProfileBackendIpcDeps> = {}) {
  const handlers = new Map<string, (event: unknown, payload: unknown) => Promise<any>>()
  const ipcMain = { handle: (channel: string, handler: any) => handlers.set(channel, handler) } as unknown as IpcMain
  const sendApplied = vi.fn()
  let config: Record<string, any> = { mode: 'local', remote: {}, profiles: {} }

  registerProfileBackendIpc(ipcMain, {
    assertCanMutateRouting: () => undefined,
    cancelAndWait: async () => undefined,
    encryptToken: token => ({ encoding: 'plain', value: token }),
    mainWindow: () => null,
    primaryProfile: () => 'default',
    profileExists: () => true,
    readConfig: () => config,
    sendApplied,
    stopPool: () => undefined,
    teardownPrimary: async () => undefined,
    teardownSsh: async () => undefined,
    writeConfig: next => {
      config = next
    },
    ...overrides
  })

  const invoke = (channel: string, payload: Record<string, unknown> = {}) =>
    handlers.get(`hermes:profile-backend:${channel}`)!({ sender: {} }, { ...LAUNCHER, ...payload })

  return {
    config: () => config,
    sendApplied,
    switchOn: async () => {
      const { grant } = await invoke('confirm-set')

      return invoke('set', { grant, url: 'http://127.0.0.1:47972', token: 'launcher-token' })
    },
    switchOff: async () => {
      const { grant } = await invoke('confirm-clear')

      return invoke('clear', { grant })
    }
  }
}

// A launcher route lives in connection.json only; the saved-gateways list beside it belongs to main and the Webapp.
describe('profile backend switch and the saved-gateways list', () => {
  it('a switch on and off writes connection.json only and leaves the list and its revision as they were', async () => {
    const stores = savedGateways()
    saveGateway(stores.main, 'Office', 'https://office.example')
    const before = fs.readFileSync(stores.file, 'utf8')
    const ipc = profileBackendIpc()

    await expect(ipc.switchOn()).resolves.toEqual({ applied: true })
    expect(ipc.config().profiles.work).toMatchObject({ mode: 'remote', launcher: { plugin: 'openshell' } })

    await expect(ipc.switchOff()).resolves.toEqual({ applied: true })
    expect(ipc.config().profiles).toEqual({})

    expect(ipc.sendApplied.mock.calls).toEqual([[{ profile: 'work' }], [{ profile: 'work' }]])
    expect(fs.readFileSync(stores.file, 'utf8')).toBe(before)
  })

  it('a failed switch restores connection.json and keeps a gateway the Webapp saved during it', async () => {
    const stores = savedGateways()
    saveGateway(stores.main, 'Office', 'https://office.example')

    const ipc = profileBackendIpc({
      // The old backend's teardown takes a while: the Webapp saves a gateway meanwhile, main's 1.5 s stat-poll
      // watcher reads the list again, and then the teardown fails.
      teardownSsh: async () => {
        saveGateway(stores.webapp, 'Lab', 'https://lab.example')
        stores.main.read()

        throw new Error('teardown failed')
      }
    })

    await expect(ipc.switchOn()).rejects.toThrow('teardown failed')

    expect(ipc.config().profiles).toEqual({})
    expect(ipc.sendApplied).not.toHaveBeenCalled()
    expect(labels(stores.webapp.read())).toContain('Lab')
    expect(labels(JSON.parse(fs.readFileSync(stores.file, 'utf8')))).toEqual(['This device', 'Office', 'Lab'])
  })
})
