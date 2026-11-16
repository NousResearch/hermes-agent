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

import type { UiPreferencesBridge } from './sync'

// Edits made from a stale view while another client changed the store
// (#145536 re-review round 3), against the real native writer on a real file:
// a removal elsewhere, an edit carried over a reload or a takeover, a tab
// whose unsent edit another tab already took over.
const MODE = 'hermes-desktop-mode-v1'
const FAVORITES = 'hermes.desktop.favorite-models'

const dirs: string[] = []
const stops: (() => void)[] = []
const listeners = new Set<(snapshot: UiPreferencesSnapshot) => void>()
const tick = (ms = 400) => new Promise(resolve => setTimeout(resolve, ms))

function storeFile(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-ui-concurrent-'))
  dirs.push(dir)

  return path.join(dir, 'ui-preferences.json')
}

const stored = (file: string, key: string) => readUiPreferences(file).values[key] ?? null
const favorites = (file: string) => JSON.parse(stored(file, FAVORITES) ?? '[]') as string[]

/** Every successful write is pushed to every other window, as native does. */
function bridgeFor(file: string, net: { online: boolean }): UiPreferencesBridge {
  const down = () => Promise.reject(new Error('fetch failed'))
  let own: ((snapshot: UiPreferencesSnapshot) => void) | null = null

  return {
    read: async () => (net.online ? readUiPreferences(file) : down()),
    write: async payload => {
      if (!net.online) {
        return down()
      }

      const snapshot = await writeUiPreferences(file, parseUiPreferencesWrite(payload))
      listeners.forEach(callback => callback !== own && callback(snapshot))

      return snapshot
    },
    onChanged: callback => {
      own = callback
      listeners.add(callback)

      return () => listeners.delete(callback)
    }
  }
}

/** One page load as entry.ts/boot.ts + main.tsx run it. */
async function pageLoad(file: string, { net = { online: true }, favoritesStore = true } = {}) {
  vi.resetModules()
  const sync = await import('./sync')
  sync.captureOwnPreferences()
  sync.hydrateFromSnapshot(net.online && fs.existsSync(file) ? readUiPreferences(file) : null)
  const store = favoritesStore ? await import('@/store/favorite-models') : null
  const storage = await import('@/lib/storage')
  stops.push(sync.startUiPreferencesSync(bridgeFor(file, net)))
  await tick()

  return { storage, store }
}

const closeTabs = () => stops.splice(0).forEach(stop => stop())

afterEach(() => {
  closeTabs()
  listeners.clear()
  vi.useRealTimers()
  window.localStorage.clear()
  window.sessionStorage.clear()

  for (const dir of dirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

it('a favorite another client removed stays removed when a tab that still lists it stars another', async () => {
  const file = storeFile()
  await writeUiPreferences(file, { values: { [FAVORITES]: JSON.stringify(['p::X', 'p::Y']) } })
  window.localStorage.setItem(FAVORITES, JSON.stringify(['p::X', 'p::Y']))
  const { store } = await pageLoad(file)

  // Native unfavorites X; the push lands here, the favorites atom keeps its read.
  const snapshot = await writeUiPreferences(file, { values: { [FAVORITES]: JSON.stringify(['p::Y']) } })
  listeners.forEach(callback => callback(snapshot))
  await tick(50)
  store?.toggleFavoriteModel('p', 'W')
  await tick()

  expect(favorites(file)).toEqual(['p::Y', 'p::W'])
})

it('an offline star carried over a reload or a takeover joins the favorites native added meanwhile', async () => {
  for (const carry of ['reload', 'takeover'] as const) {
    const file = storeFile()
    await writeUiPreferences(file, { values: { [FAVORITES]: JSON.stringify(['p::A']) } })
    window.localStorage.setItem(FAVORITES, JSON.stringify(['p::A']))
    await pageLoad(file)
    closeTabs()

    const { store } = await pageLoad(file, { net: { online: false } })
    store?.toggleFavoriteModel('p', 'W')
    await tick()

    if (carry === 'takeover') {
      window.dispatchEvent(new PageTransitionEvent('pagehide'))
      window.sessionStorage.clear()
    }

    closeTabs()
    await writeUiPreferences(file, { values: { [FAVORITES]: JSON.stringify(['p::A', 'p::N']) } })
    await pageLoad(file)

    expect([carry, favorites(file).sort()]).toEqual([carry, ['p::A', 'p::N', 'p::W']])
    closeTabs()
    window.localStorage.clear()
    window.sessionStorage.clear()
  }
}, 15_000)

it('a frozen tab whose unsent edit another tab took over never re-sends it over a newer choice', async () => {
  const file = storeFile()
  await writeUiPreferences(file, { values: { [MODE]: 'light' } })
  window.localStorage.setItem(MODE, 'light')
  const frozenNet = { online: true }
  const frozen = await pageLoad(file, { favoritesStore: false, net: frozenNet })
  frozenNet.online = false
  frozen.storage.writeKey(MODE, 'dark')
  await tick()

  // That tab stops beating for two minutes; a new tab adopts its edit, then
  // the user picks again there.
  vi.useFakeTimers({ now: Date.now() + 120_000, toFake: ['Date'] })
  window.sessionStorage.clear()
  const fresh = await pageLoad(file, { favoritesStore: false })
  fresh.storage.writeKey(MODE, 'system')
  await tick()
  vi.useRealTimers()

  frozenNet.online = true // the frozen tab wakes; its retry fires
  await tick(6000)

  expect(stored(file, MODE)).toBe('system')
}, 15_000)
