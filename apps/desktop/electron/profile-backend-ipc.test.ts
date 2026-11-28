import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import type { IpcMain } from 'electron'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { BackendDialClaims, DIAL_ROUTE_REPLACED } from './backend-dial-claim'
import { resolveProfileBackendRoute } from './connection-config'
import { sanitizeConnectionProfiles } from './connection-profiles'
import {
  backendScopeKey,
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
import { primaryDialClaimKey } from './primary-boot-join'
import { type ProfileBackendIpcDeps, registerProfileBackendIpc } from './profile-backend-ipc'

// The user answers every native confirm with its first button (Allow) unless a test holds the dialog open.
const electron = vi.hoisted(() => ({
  showMessageBox: vi.fn(async (): Promise<{ response: number }> => ({ response: 0 })),
  windows: [] as {
    isDestroyed: () => boolean
    webContents: { isDestroyed: () => boolean; send: (...args: unknown[]) => void }
  }[]
}))

vi.mock('electron', () => ({
  BrowserWindow: { fromWebContents: () => null, getAllWindows: () => electron.windows },
  dialog: { showMessageBox: electron.showMessageBox }
}))

// A chat window as main sees it; `sent` is what its renderer was told.
function chatWindow({ destroyed = false } = {}) {
  const send = vi.fn()
  electron.windows.push({ isDestroyed: () => destroyed, webContents: { isDestroyed: () => destroyed, send } })

  return { sent: () => send.mock.calls }
}

const LAUNCHER = { plugin: 'openshell', profile: 'work' }

const dirs: string[] = []

afterEach(() => {
  electron.windows.length = 0
  electron.showMessageBox.mockReset().mockImplementation(async () => ({ response: 0 }))

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
function profileBackendIpc(overrides: Partial<ProfileBackendIpcDeps> = {}, profiles: Record<string, any> = {}) {
  const handlers = new Map<string, (event: unknown, payload: unknown) => Promise<any>>()
  const ipcMain = { handle: (channel: string, handler: any) => handlers.set(channel, handler) } as unknown as IpcMain
  let config: Record<string, any> = { mode: 'local', remote: {}, profiles }

  registerProfileBackendIpc(ipcMain, {
    assertCanMutateRouting: () => undefined,
    cancelAndWait: async () => undefined,
    encryptToken: token => ({ encoding: 'plain', value: token }),
    forgetDial: () => undefined,
    mainWindow: () => null,
    primaryProfile: () => 'default',
    profileExists: () => true,
    // connection.json is read back through the sanitizer (main's readDesktopConnectionConfig).
    readConfig: () => ({ ...config, profiles: sanitizeConnectionProfiles(config.profiles) }),
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
    invoke,
    switchOn: async (payload: Record<string, unknown> = {}) => {
      const { grant } = await invoke('confirm-set', payload)

      return invoke('set', { ...payload, grant, url: 'http://127.0.0.1:47972', token: 'launcher-token' })
    },
    switchOff: async () => {
      const { grant } = await invoke('confirm-clear')

      return invoke('clear', { grant })
    },
    // Another writer of connection.json (a Settings apply, another plugin's switch).
    writeProfile: (profile: string, entry: Record<string, unknown>) => {
      config = { ...config, profiles: { ...config.profiles, [profile]: entry } }
    }
  }
}

// A confirm the user has not answered yet; `answer` clicks its first button.
function openConfirm() {
  let answer = () => undefined as void

  electron.showMessageBox.mockImplementationOnce(
    () =>
      new Promise(resolve => {
        answer = () => resolve({ response: 0 })
      })
  )

  return { answer: () => answer(), shown: () => vi.waitFor(() => expect(electron.showMessageBox).toHaveBeenCalled()) }
}

// Main's two caches of a profile's backend, as the `hermes:connection` handler and ensureBackend use them: a dial in
// flight answers every caller of its claim key, and a pool entry answers every later dial of its profile.
function backendDials() {
  const claims = new BackendDialClaims()
  const pool = new Map<string, Promise<string>>()

  const dial = (profile: string, start: () => Promise<string>, inheritsPrimary = false) =>
    claims.run(primaryDialClaimKey(backendScopeKey(null, profile), inheritsPrimary), () => {
      const pooled = pool.get(profile) ?? start()

      pool.set(profile, pooled)

      return pooled
    })

  return {
    deps: { forgetDial: (key: string) => claims.forget(key), stopPool: (scope: string) => pool.delete(scope) },
    dial
  }
}

// A dial waiting on a backend that never answers: main waits minutes for a remote address to become ready.
const neverReady = () => new Promise<string>(() => undefined)

// A launcher route lives in connection.json only; the saved-gateways list beside it belongs to main and the Webapp.
describe('profile backend switch and the saved-gateways list', () => {
  it('a switch on and off writes connection.json only and leaves the list and its revision as they were', async () => {
    const stores = savedGateways()
    saveGateway(stores.main, 'Office', 'https://office.example')
    const before = fs.readFileSync(stores.file, 'utf8')
    const win = chatWindow()
    const ipc = profileBackendIpc()

    await expect(ipc.switchOn()).resolves.toEqual({ applied: true })
    expect(ipc.config().profiles.work).toMatchObject({ mode: 'remote', launcher: { plugin: 'openshell' } })

    await expect(ipc.switchOff()).resolves.toEqual({ applied: true })
    expect(ipc.config().profiles).toEqual({})

    expect(win.sent()).toEqual([
      ['hermes:connection:applied', { profile: 'work' }],
      ['hermes:connection:applied', { profile: 'work' }]
    ])
    expect(fs.readFileSync(stores.file, 'utf8')).toBe(before)
  })

  it('a failed switch restores connection.json and keeps a gateway the Webapp saved during it', async () => {
    const stores = savedGateways()
    saveGateway(stores.main, 'Office', 'https://office.example')
    const win = chatWindow()

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
    expect(win.sent()).toEqual([])
    expect(labels(stores.webapp.read())).toContain('Lab')
    expect(labels(JSON.parse(fs.readFileSync(stores.file, 'utf8')))).toEqual(['This device', 'Office', 'Lab'])
  })
})

describe('profile backend switch and the other routes in connection.json', () => {
  const SAVED_SSH = { mode: 'ssh', host: 'box.example', user: 'me' }
  const REMOTE = { mode: 'remote', url: 'https://box.example', token: { encoding: 'plain', value: 't' } }

  it('a switch off puts back the local entry the switch on replaced', async () => {
    const ipc = profileBackendIpc({}, { work: { mode: 'local', savedSsh: SAVED_SSH } })

    await ipc.switchOn()
    expect(ipc.config().profiles.work).toMatchObject({ mode: 'remote', launcher: { plugin: 'openshell' } })

    await ipc.switchOff()
    expect(ipc.config().profiles.work).toMatchObject({ mode: 'local', savedSsh: SAVED_SSH })

    // With its own entry the profile stays on this computer when the primary profile's backend is remote.
    expect(
      resolveProfileBackendRoute('work', {
        ownEntry: Boolean(ipc.config().profiles.work),
        primaryProfile: 'default',
        primaryRemoteActive: true
      }).backend
    ).toBe('pool')
  })

  it('a switch off of a profile that had no entry leaves none', async () => {
    const ipc = profileBackendIpc()

    await ipc.switchOn()
    await ipc.switchOff()

    expect(ipc.config().profiles).toEqual({})
  })

  it("a failed switch puts back only its own profile's entry", async () => {
    let failWork = (_error: Error) => undefined as void

    const ipc = profileBackendIpc({
      teardownSsh: scope =>
        scope === 'work'
          ? new Promise<void>((_resolve, reject) => {
              failWork = reject
            })
          : Promise.resolve()
    })

    // The old backend of "work" is still being retired when another profile's switch completes.
    const work = ipc.switchOn()

    await vi.waitFor(() => expect(ipc.config().profiles.work).toBeDefined())
    await expect(ipc.switchOn({ profile: 'notes' })).resolves.toEqual({ applied: true })

    failWork(new Error('teardown failed'))
    await expect(work).rejects.toThrow('teardown failed')

    expect(ipc.config().profiles.work).toBeUndefined()
    expect(ipc.config().profiles.notes).toMatchObject({ launcher: { plugin: 'openshell' } })
  })

  it('a failed switch leaves a route written to its profile after its own', async () => {
    let failWork = (_error: Error) => undefined as void

    const ipc = profileBackendIpc({
      teardownSsh: () =>
        new Promise<void>((_resolve, reject) => {
          failWork = reject
        })
    })

    const work = ipc.switchOn()

    await vi.waitFor(() => expect(ipc.config().profiles.work).toBeDefined())
    ipc.writeProfile('work', REMOTE)
    failWork(new Error('teardown failed'))
    await expect(work).rejects.toThrow('teardown failed')

    expect(ipc.config().profiles.work).toEqual(REMOTE)
  })

  it('a set whose confirm stayed open checks again who routes the profile', async () => {
    const ipc = profileBackendIpc()
    const confirm = openConfirm()
    // No grant: the set itself asks. Meanwhile the user connects the profile to a remote host.
    const set = ipc.invoke('set', { url: 'http://127.0.0.1:47972', token: 'launcher-token' })

    await confirm.shown()
    ipc.writeProfile('work', REMOTE)
    confirm.answer()

    await expect(set).rejects.toThrow('Profile "work" already connects to a remote host.')
    expect(ipc.config().profiles.work).toEqual(REMOTE)
  })

  it('a clear whose confirm stayed open checks again who routes the profile', async () => {
    const ipc = profileBackendIpc()
    const other = { mode: 'remote', url: 'http://127.0.0.1:5', token: REMOTE.token, launcher: { plugin: 'other' } }

    await ipc.switchOn()

    const confirm = openConfirm()
    const clear = ipc.invoke('clear')

    await confirm.shown()
    ipc.writeProfile('work', other)
    confirm.answer()

    await expect(clear).rejects.toThrow('Profile "work" runs through the other plugin.')
    expect(ipc.config().profiles.work).toEqual(other)
  })
})

describe('profile backend switch and the windows', () => {
  it('tells every open window that the profile moved, not only the main one', async () => {
    const main = chatWindow()
    const peer = chatWindow()
    const closing = chatWindow({ destroyed: true })
    const ipc = profileBackendIpc()

    await ipc.switchOn()
    await ipc.switchOff()

    const moved = ['hermes:connection:applied', { profile: 'work' }]

    expect(main.sent()).toEqual([moved, moved])
    expect(peer.sent()).toEqual([moved, moved])
    expect(closing.sent()).toEqual([])
  })
})

// While a sandbox stops, its launcher is gone seconds before the route is cleared. A window that lost its socket asks
// main for the profile's backend in that gap, and main then waits on the launcher's address for its whole readiness
// limit (three minutes).
describe('profile backend switch and the dials in flight', () => {
  it.each([false, true])(
    'a dial still waiting on the stopped launcher does not answer the dials made after the switch off (primary: %s)',
    async primary => {
      const dials = backendDials()
      const ipc = profileBackendIpc({ ...dials.deps, primaryProfile: () => (primary ? 'work' : 'default') })

      await ipc.switchOn()

      const waiting = dials.dial('work', neverReady, primary)

      await ipc.switchOff()
      // The window whose dial that was is told at once, so its reconnect asks again.
      await expect(waiting).rejects.toThrow(DIAL_ROUTE_REPLACED)

      // The windows were told the profile moved and dial again: this computer's backend serves it now.
      const host = vi.fn(async () => 'this computer')
      const answered = dials.dial('work', host, primary)

      expect(host).toHaveBeenCalledTimes(1)
      await expect(answered).resolves.toBe('this computer')
    }
  )

  it('a dial made while the old route is torn down reads the new route, and keeps its place', async () => {
    const dials = backendDials()
    const host = vi.fn(async () => 'this computer')
    let stopping = false
    let duringTeardown: Promise<string> | undefined

    const ipc = profileBackendIpc({
      ...dials.deps,
      // The teardown is awaited after connection.json is written: a window's reconnect timer can fire in between.
      teardownSsh: async () => {
        if (stopping) {
          duringTeardown = dials.dial('work', host)
        }
      }
    })

    await ipc.switchOn()
    dials.dial('work', neverReady).catch(() => undefined)
    stopping = true
    await ipc.switchOff()

    expect(host).toHaveBeenCalledTimes(1)
    await expect(duringTeardown).resolves.toBe('this computer')

    // The backend that dial found stays the profile's: the switch does not drop it a second time.
    const again = vi.fn(neverReady)
    const later = dials.dial('work', again)

    expect(again).not.toHaveBeenCalled()
    await expect(later).resolves.toBe('this computer')
  })

  it('a failed switch off leaves nothing dialed for the route it took back', async () => {
    const dials = backendDials()
    let failing = false

    const ipc = profileBackendIpc({
      ...dials.deps,
      teardownSsh: async () => {
        if (failing) {
          // A window dialed the cleared route before the teardown failed.
          dials.dial('work', async () => 'this computer').catch(() => undefined)

          throw new Error('teardown failed')
        }
      }
    })

    await ipc.switchOn()
    failing = true
    await expect(ipc.switchOff()).rejects.toThrow('teardown failed')

    // The launcher's route is back in connection.json: the next dial reads it.
    const launcher = vi.fn(async () => 'launcher')
    const answered = dials.dial('work', launcher)

    expect(launcher).toHaveBeenCalledTimes(1)
    await expect(answered).resolves.toBe('launcher')
  })
})
