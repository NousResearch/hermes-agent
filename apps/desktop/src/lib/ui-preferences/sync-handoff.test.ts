import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, it, vi } from 'vitest'

import {
  parseUiPreferencesWrite,
  readUiPreferences,
  writeUiPreferences
} from '../../../electron/ui-preferences'

import type { UiPreferencesBridge } from './sync'

// Hand-offs between clients and page loads, against the real native writer on
// a real file (#145536 re-review): an upgraded browser's old defaults, a closed
// tab's unsent edit, a first snapshot that lands after the app read its cache.
const KEYBINDS = 'hermes.desktop.keybinds'
const HAPTICS = 'hermes.desktop.hapticsMuted'
const MODE = 'hermes-desktop-mode-v1'
const FAVORITES = 'hermes.desktop.favorite-models'
const OUTBOX_PREFIX = 'hermes:ui-preferences:outbox:'
const CUSTOM_KEYBINDS = '{"composer.dictate":["mod+shift+d"]}'

// Every store that writes its value the moment its module evaluates.
const LOAD_WRITERS = [
  () => import('@/store/backdrop'),
  () => import('@/store/completion-sound'),
  () => import('@/store/external-links'),
  () => import('@/store/haptics'),
  () => import('@/store/keybinds'),
  () => import('@/store/profile'),
  () => import('@/store/reasoning-disclosure'),
  () => import('@/store/thread-timeline'),
  () => import('@/store/tool-view'),
  () => import('@/store/translucency')
]

const dirs: string[] = []
const stops: (() => void)[] = []
const tick = (ms = 400) => new Promise(resolve => setTimeout(resolve, ms))

function storeFile(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-ui-handoff-'))
  dirs.push(dir)

  return path.join(dir, 'ui-preferences.json')
}

function bridgeFor(file: string, { offline = false, slow = false } = {}): UiPreferencesBridge {
  const down = () => Promise.reject(new Error('fetch failed'))

  return {
    read: async () => (offline ? down() : (slow && (await tick(300)), readUiPreferences(file))),
    write: async payload => (offline ? down() : writeUiPreferences(file, parseUiPreferencesWrite(payload))),
    onChanged: () => () => undefined
  }
}

interface LoadOptions {
  /** No snapshot before the app loads: the boot read timed out or failed. */
  late?: boolean
  offline?: boolean
  stores?: (() => Promise<unknown>)[]
}

/** One page load as entry.ts/boot.ts + main.tsx run it. */
async function pageLoad(file: string, { late = false, offline = false, stores = LOAD_WRITERS }: LoadOptions = {}) {
  vi.resetModules()
  const sync = await import('./sync')
  sync.captureOwnPreferences()
  sync.hydrateFromSnapshot(offline || late || !fs.existsSync(file) ? null : readUiPreferences(file))

  for (const load of stores) {
    await load()
  }

  stops.push(sync.startUiPreferencesSync(bridgeFor(file, { offline, slow: late })))
  await tick(late ? 700 : 400)
}

function closeTabs() {
  stops.splice(0).forEach(stop => stop())
}

/** This origin's storage is another client's: native Desktop, or a new browser profile. */
function becomeClient(values: Record<string, string>) {
  closeTabs()
  window.localStorage.clear()
  window.sessionStorage.clear()
  Object.entries(values).forEach(([key, value]) => window.localStorage.setItem(key, value))
}

const outboxes = () =>
  Array.from({ length: window.localStorage.length }, (_, index) => window.localStorage.key(index) ?? '').filter(key =>
    key.startsWith(OUTBOX_PREFIX)
  )

afterEach(() => {
  closeTabs()
  vi.useRealTimers()
  window.localStorage.clear()
  window.sessionStorage.clear()

  for (const dir of dirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

it('a browser upgraded from a build that stored every default at load never seeds them over native’s choices', async () => {
  const file = storeFile()

  // What that build left in a browser: each load-time store's default, unmarked.
  await pageLoad(file, { offline: true })

  const oldDefaults = Object.fromEntries(
    Array.from({ length: window.localStorage.length }, (_, index) => window.localStorage.key(index) ?? '')
      .filter(key => key.startsWith('hermes.desktop.') || key.startsWith('hermes-desktop-'))
      .map(key => [key, window.localStorage.getItem(key) ?? ''])
  )

  expect(oldDefaults).toMatchObject({ [HAPTICS]: 'false', [KEYBINDS]: '{}' })

  becomeClient(oldDefaults)
  await pageLoad(file)

  becomeClient({ [HAPTICS]: 'true', [KEYBINDS]: CUSTOM_KEYBINDS })
  await pageLoad(file)

  expect([window.localStorage.getItem(KEYBINDS), window.localStorage.getItem(HAPTICS)]).toEqual([
    CUSTOM_KEYBINDS,
    'true'
  ])
  expect(readUiPreferences(file).values).toEqual({ [HAPTICS]: 'true', [KEYBINDS]: CUSTOM_KEYBINDS })
})
