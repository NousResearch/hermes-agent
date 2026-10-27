import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, it, vi } from 'vitest'

import {
  parseUiPreferencesWrite,
  readUiPreferences,
  type UiPreferencesSnapshot,
  writeUiPreferences
} from '../../../electron/ui-preferences'

import { hydrateFromSnapshot, startUiPreferencesSync, type UiPreferencesBridge } from './sync'

// The renderer sync against the real native writer on a real file (#145536 review).
const DENSITY = 'hermes.desktop.sessionListDensity'
const MODE = 'hermes-desktop-mode-v1'
const THEME = 'hermes-desktop-theme-v2'
const USER_THEMES = 'hermes-desktop-user-themes-v1'
const KEYBINDS = 'hermes.desktop.keybinds'
const dirs: string[] = []
const tick = (ms = 400) => new Promise(resolve => setTimeout(resolve, ms))

function storeFile(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-ui-sync-'))
  dirs.push(dir)

  return path.join(dir, 'ui-preferences.json')
}

function realBridge(file: string) {
  let changed: ((snapshot: UiPreferencesSnapshot) => void) | null = null

  const bridge: UiPreferencesBridge = {
    read: async () => readUiPreferences(file),
    write: payload => writeUiPreferences(file, parseUiPreferencesWrite(payload)),
    onChanged: callback => {
      changed = callback

      return () => (changed = null)
    }
  }

  return { bridge, push: () => changed?.(readUiPreferences(file)) }
}

afterEach(() => {
  window.localStorage.clear()
  window.sessionStorage.clear()

  for (const dir of dirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

it('a store a fresh client created never removes another client’s values: it imports them', async () => {
  const file = storeFile()
  // A fresh browser (no shared keys yet) makes first contact.
  let stop = startUiPreferencesSync(realBridge(file).bridge)
  await tick()
  stop()

  // Native Desktop, another origin, launches with its own preferences.
  window.localStorage.clear()
  const mine = { [KEYBINDS]: '{"newChat":"mod+n"}', [THEME]: 'nous', [USER_THEMES]: '{"mine":{"name":"Mine"}}' }
  Object.entries(mine).forEach(([key, value]) => window.localStorage.setItem(key, value))
  hydrateFromSnapshot(readUiPreferences(file))
  stop = startUiPreferencesSync(realBridge(file).bridge)
  await tick()
  stop()

  expect(Object.fromEntries(Object.keys(mine).map(key => [key, window.localStorage.getItem(key)]))).toEqual(mine)
  expect(readUiPreferences(file).values).toEqual(mine)
})

it('a corrupt store is kept aside and open windows follow its replacement, re-importing what it lost', async () => {
  const file = storeFile()
  await writeUiPreferences(file, { values: { [MODE]: 'dark' } })

  for (let n = 0; n < 5; n += 1) {
    await writeUiPreferences(file, { values: { [DENSITY]: `d${n}` } })
  }

  const { bridge, push } = realBridge(file)
  const stop = startUiPreferencesSync(bridge)
  await tick()
  expect(window.localStorage.getItem(DENSITY)).toBe('d4')

  const torn = '{"version":1,"revision":6,"val'
  fs.writeFileSync(file, torn)
  await writeUiPreferences(file, { values: { [DENSITY]: 'compact' } }) // e.g. the Webapp's next write
  push()
  await tick()
  stop()

  const backups = fs.readdirSync(path.dirname(file)).filter(name => name.startsWith('ui-preferences.json.corrupt-'))

  expect(window.localStorage.getItem(DENSITY)).toBe('compact')
  expect(readUiPreferences(file).values).toEqual({ [DENSITY]: 'compact', [MODE]: 'dark' })
  expect(backups.map(name => fs.readFileSync(path.join(path.dirname(file), name), 'utf8'))).toEqual([torn])
})

it('same-origin tabs keep each other’s unsent changes', async () => {
  const file = storeFile()
  await writeUiPreferences(file, { values: { [DENSITY]: 'compact', [MODE]: 'light' } })
  const offline: UiPreferencesBridge = {
    read: () => new Promise(() => undefined),
    write: () => new Promise(() => undefined)
  }

  // Two tabs of one origin (one localStorage, a sessionStorage each), offline.
  vi.resetModules()
  const tabA = { storage: await import('../storage'), sync: await import('./sync') }
  const stopA = tabA.sync.startUiPreferencesSync(offline)
  const sessionA = Object.fromEntries(Object.entries(window.sessionStorage))
  window.sessionStorage.clear()
  vi.resetModules()
  const tabB = { storage: await import('../storage'), sync: await import('./sync') }
  const stopB = tabB.sync.startUiPreferencesSync(offline)

  tabA.storage.writeKey(DENSITY, 'detailed')
  tabB.storage.writeKey(MODE, 'dark')
  stopA()
  stopB()

  // Tab A reloads once the store is reachable again.
  window.sessionStorage.clear()
  Object.entries(sessionA).forEach(([key, value]) => window.sessionStorage.setItem(key, value))
  vi.resetModules()
  const reloadedA = await import('./sync')
  const stop = reloadedA.startUiPreferencesSync(realBridge(file).bridge)
  await tick()
  stop()

  expect(readUiPreferences(file).values[DENSITY]).toBe('detailed')
})
