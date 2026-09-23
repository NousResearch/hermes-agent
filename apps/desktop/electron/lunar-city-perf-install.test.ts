import { EventEmitter } from 'node:events'

import { afterEach, expect, test, vi } from 'vitest'

vi.mock('electron', async () => {
  const { EventEmitter } = await import('node:events')

  const ipcMain = Object.assign(new EventEmitter(), {
    handle: vi.fn(),
    removeHandler: vi.fn()
  })

  return { app: { isPackaged: true }, ipcMain, BrowserWindow: {}, contentTracing: {}, screen: {}, webContents: {} }
})

import { type BrowserWindow, ipcMain } from 'electron'

import { installLunarCityPerfBridge } from './lunar-city-perf-install'

afterEach(() => vi.unstubAllEnvs())

test('installed IPC binds authority to the owned renderer and revokes it on navigation and disposal', () => {
  vi.stubEnv('HERMES_LUNAR_CITY_PERF_ACCEPTANCE', '1')
  vi.stubEnv('HERMES_LUNAR_CITY_PERF_NONCE', 'nonce-0123456789abcdef-unique')
  const sender = Object.assign(new EventEmitter(), { id: 41, isDestroyed: () => false, getOSProcessId: () => 82 })
  const window = { webContents: sender, isDestroyed: () => false } as unknown as BrowserWindow

  const bridge = installLunarCityPerfBridge({
    buildStamp: {
      builtAt: '2026-08-31T12:00:00.000Z',
      commit: 'a'.repeat(40),
      dirty: false,
      schemaVersion: 1,
      source: 'local'
    },
    getMainWindow: () => window
  })

  bridge.attachWindow(window)

  const request = (channel: string, value?: unknown, requestSender = sender) => {
    const event = { sender: requestSender, senderFrame: { routingId: 0 }, returnValue: undefined as unknown }
    ipcMain.emit(`hermes:lunar-city-perf:${channel}`, event, value)

    return event.returnValue
  }

  try {
    const handshake = request('bootstrap')
    expect(handshake).toMatchObject({ rendererIdentity: { pid: 82 }, packaged: true })
    expect(request('bootstrap', undefined, Object.assign(new EventEmitter(), { ...sender, id: 999 }))).toBeUndefined()
    expect(request('register-responder', handshake)).toBe(true)
    expect(request('activate', handshake)).toBe(true)
    sender.emit('did-start-navigation', {}, 'file:///reload', false, true)
    expect(request('activate', handshake)).toBe(false)
  } finally {
    bridge.dispose()
  }

  expect(sender.listenerCount('did-start-navigation')).toBe(0)
  expect(sender.listenerCount('render-process-gone')).toBe(0)
  expect(ipcMain.listenerCount('hermes:lunar-city-perf:bootstrap')).toBe(0)
  expect(ipcMain.removeHandler).toHaveBeenCalledWith('hermes:lunar-city-perf:process-metrics')
  expect(request('bootstrap')).toBeUndefined()
})
