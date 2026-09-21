import { EventEmitter } from 'node:events'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, it, vi } from 'vitest'

const native = vi.hoisted(() => ({ start: vi.fn(), stop: vi.fn() }))
const state = vi.hoisted(() => ({ directory: '', windows: [] as any[], handlers: new Map<string, any>() }))
vi.mock('./hud-modifier-monitor', () => ({
  HudModifierMonitor: class {
    start = native.start
    stop = native.stop
  }
}))
vi.mock('electron', async () => {
  const { EventEmitter } = await import('node:events')

  return {
    app: Object.assign(new EventEmitter(), { getPath: () => state.directory, getAppPath: () => '/app' }),
    powerMonitor: new EventEmitter(),
    BrowserWindow: {
      getAllWindows: () => state.windows,
      fromWebContents: (wc: unknown) => state.windows.find(win => win.webContents === wc)
    },
    ipcMain: {
      handle: (name: string, handler: unknown) => state.handlers.set(name, handler),
      removeHandler: (name: string) => state.handlers.delete(name)
    },
    shell: { openExternal: vi.fn() }
  }
})

import { app, powerMonitor } from 'electron'

import { installHudModifierTap } from './hud-modifier'

const cleanups: (() => void)[] = []
afterEach(async () => {
  cleanups.splice(0).forEach(dispose => dispose())
  await rm(state.directory, { recursive: true, force: true })
  state.windows = []
  vi.clearAllMocks()
})

async function setup() {
  state.directory = await mkdtemp(path.join(os.tmpdir(), 'hud-modifier-settings-'))
  const summon = vi.fn()
  const frame = { url: 'http://127.0.0.1:5174/?win=hud#/' }
  const wc = Object.assign(new EventEmitter(), { mainFrame: frame, send: vi.fn() })
  const win = { webContents: wc, isDestroyed: () => false }
  state.windows.push(win)
  const event = { sender: wc, senderFrame: frame }

  const install = () => {
    const dispose = installHudModifierTap({ rendererUrl: 'http://127.0.0.1:5174', summon })
    cleanups.push(dispose)

    return dispose
  }

  const call = (name: string, value?: unknown, sender = event) =>
    state.handlers.get(`hermes:hud-modifier:${name}`)(sender, value)

  return { summon, event, call, install }
}

it('persists opt-in, fences stale native callbacks and re-arms after lock without requesting permission', async () => {
  const { summon, call, install } = await setup()
  const dispose = install()
  expect(await call('settings:get')).toEqual({ enabled: false, state: 'disabled' })
  expect(native.start).not.toHaveBeenCalled()
  await call('settings:set', true)
  expect(JSON.parse(await readFile(path.join(state.directory, 'hud-modifier.json'), 'utf8'))).toEqual({ enabled: true })
  const [oldTap, oldStatus, permission] = native.start.mock.calls.at(-1)!
  expect(permission).toBe(true)
  oldTap()
  expect(summon).not.toHaveBeenCalled()
  oldStatus({ type: 'ready' })
  oldTap()
  expect(summon).toHaveBeenCalledOnce()
  powerMonitor.emit('lock-screen')
  oldTap()
  oldStatus({ type: 'ready' })
  expect(summon).toHaveBeenCalledOnce()
  powerMonitor.emit('unlock-screen')
  expect(native.start.mock.calls.at(-1)![2]).toBe(false)
  await call('settings:set', false)
  const [lateTap, lateStatus] = native.start.mock.calls.at(-1)!
  lateStatus({ type: 'ready' })
  lateTap()
  expect(await call('settings:get')).toEqual({ enabled: false, state: 'disabled' })
  expect(summon).toHaveBeenCalledOnce()
  await call('settings:set', true)
  dispose()
  install()
  expect(native.start.mock.calls.at(-1)![2]).toBe(false)
  const [restartTap, restartStatus] = native.start.mock.calls.at(-1)!
  for (const reason of ['missing-helper', 'unsupported-session']) {
    restartStatus({ type: 'error', code: 'unavailable', reason })
    expect(await call('settings:get')).toEqual({ enabled: true, state: 'unavailable', reason })
  }
  restartStatus({ type: 'ready' })
  expect(await call('settings:get')).toEqual({ enabled: true, state: 'ready' })
  app.emit('will-quit')
  restartTap()
  expect(summon).toHaveBeenCalledOnce()
  expect(state.handlers.size).toBe(0)
  expect(powerMonitor.listenerCount('unlock-screen')).toBe(0)
})

it('rejects guest frames and invalid writes, and reports native permission failure without summoning', async () => {
  const { summon, event, call, install } = await setup()
  install()
  await expect(call('settings:set', true, { ...event, senderFrame: { url: 'https://example.org/' } })).rejects.toThrow(
    'untrusted'
  )
  await expect(call('settings:set', 'true')).rejects.toThrow('boolean')
  expect(native.start).not.toHaveBeenCalled()
  await call('settings:set', true)
  const [tap, status] = native.start.mock.calls.at(-1)!
  status({ type: 'error', code: 'permission-required' })
  tap()
  expect(await call('settings:get')).toEqual({ enabled: true, state: 'input-permission' })
  expect(summon).not.toHaveBeenCalled()
  // A failed authoritative write does not silently disable a live setting.
  const directory = state.directory
  await rm(directory, { recursive: true, force: true })
  const { writeFile } = await import('node:fs/promises')
  await writeFile(directory, 'not a directory')
  await expect(call('settings:set', false)).rejects.toThrow()
  expect(await call('settings:get')).toEqual({ enabled: true, state: 'input-permission' })
})
