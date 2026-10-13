import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, expect, test, vi } from 'vitest'

const native = vi.hoisted(() => ({
  windows: [] as any[],
  ipc: new Map<string, (...args: any[]) => any>()
}))

vi.mock('electron', () => ({
  BrowserWindow: { getAllWindows: () => native.windows },
  ipcMain: { handle: (name: string, fn: (...args: any[]) => any) => native.ipc.set(name, fn) }
}))

import { createDownloadSavePrefs, unclaimedDownloadPath } from './download-save-prefs'

let dir: string
let preferencesPath: string

class Window {
  sent: unknown[] = []
  destroyed = false
  webContents = { send: (_channel: string, value: unknown) => this.sent.push(value) }
  isDestroyed() {
    return this.destroyed
  }
}

beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), 'download-save-'))
  preferencesPath = path.join(dir, 'download-save-direct.json')
  native.windows.length = 0
  native.ipc.clear()
})

afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true })
  vi.clearAllMocks()
})

test('start keeps the save dialog when the preference is missing or malformed', () => {
  const prefs = createDownloadSavePrefs({ preferencesPath })

  prefs.start()

  expect(prefs.isEnabled()).toBe(false)

  fs.writeFileSync(preferencesPath, '{not json', 'utf8')
  prefs.start()

  expect(prefs.isEnabled()).toBe(false)
})

test('start adopts a persisted direct-save preference', () => {
  fs.writeFileSync(preferencesPath, JSON.stringify({ direct: true }), 'utf8')

  const prefs = createDownloadSavePrefs({ preferencesPath })

  prefs.start()

  expect(prefs.isEnabled()).toBe(true)
})

test('setDirect persists atomically, flips the getter, and broadcasts to live windows', () => {
  const prefs = createDownloadSavePrefs({ preferencesPath })
  const live = new Window()
  const destroyed = new Window()
  destroyed.destroyed = true
  native.windows.push(live, destroyed)

  prefs.start()

  expect(prefs.setDirect(true)).toBe(true)

  expect(JSON.parse(fs.readFileSync(preferencesPath, 'utf8'))).toEqual({ direct: true })
  expect(prefs.isEnabled()).toBe(true)
  expect(live.sent).toEqual([true])
  expect(destroyed.sent).toEqual([])
  expect(fs.existsSync(`${preferencesPath}.tmp`)).toBe(false)
})

test('setDirect keeps the session flip even when the disk write fails', () => {
  const log = vi.fn()
  const prefs = createDownloadSavePrefs({ preferencesPath, log })

  const writeSpy = vi.spyOn(fs, 'writeFileSync').mockImplementation(() => {
    throw new Error('disk full')
  })

  prefs.start()

  expect(prefs.setDirect(true)).toBe(true)
  expect(prefs.isEnabled()).toBe(true)
  expect(log).toHaveBeenCalledWith(expect.stringContaining('disk full'))

  writeSpy.mockRestore()
})

test('IPC handlers read and write the preference', () => {
  const prefs = createDownloadSavePrefs({ preferencesPath })

  prefs.start()

  expect(native.ipc.get('hermes:download-save-direct:get')!()).toBe(false)
  expect(native.ipc.get('hermes:download-save-direct:set')!(undefined, true)).toBe(true)
  expect(native.ipc.get('hermes:download-save-direct:get')!()).toBe(true)
})

test('unclaimedDownloadPath returns the plain target when it is free', () => {
  expect(unclaimedDownloadPath('/downloads', 'report.pdf', () => false)).toBe('/downloads/report.pdf')
})

test('unclaimedDownloadPath never resolves onto an existing file', () => {
  const taken = new Set(['/downloads/report.pdf', '/downloads/report (1).pdf'])

  expect(unclaimedDownloadPath('/downloads', 'report.pdf', candidate => taken.has(candidate))).toBe(
    '/downloads/report (2).pdf'
  )
  // Extensionless names (the pane's uuid.tmp case) keep their stem intact.
  expect(unclaimedDownloadPath('/downloads', 'abc.tmp', candidate => taken.has(candidate))).toBe('/downloads/abc.tmp')
})
