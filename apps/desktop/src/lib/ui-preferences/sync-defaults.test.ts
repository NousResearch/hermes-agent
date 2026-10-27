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

// A value a store writes for itself at load (its default, a migration from
// config) is not the user's choice and must never reach the machine's store
// ahead of the client that holds the real one (#145536 review), against the
// real native writer on a real file.
const TRANSLUCENCY = 'hermes.desktop.translucency.v2'
const AUTO_SPEAK = 'hermes.desktop.autoSpeakReplies'
const THEME = 'hermes-desktop-theme-v2'
const dirs: string[] = []
const tick = (ms = 400) => new Promise(resolve => setTimeout(resolve, ms))

function storeFile(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-ui-defaults-'))
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

/** One page load, as boot.ts + main.tsx run it: capture, hydrate, then the app's stores. */
async function load(file: string, app: () => Promise<void>) {
  vi.resetModules()
  const sync = await import('./sync')
  sync.captureOwnPreferences()
  sync.hydrateFromSnapshot(readUiPreferences(file))
  const { bridge, push } = realBridge(file)
  const stop = sync.startUiPreferencesSync(bridge)
  await app()
  await tick()

  return { push, stop }
}

/** Native Desktop's localStorage is another origin's: switch to it. */
function becomeNative(values: Record<string, string>) {
  window.localStorage.clear()
  window.sessionStorage.clear()
  Object.entries(values).forEach(([key, value]) => window.localStorage.setItem(key, value))
}

afterEach(() => {
  window.localStorage.clear()
  window.sessionStorage.clear()

  for (const dir of dirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

it('a fresh browser’s translucency default never replaces native’s custom translucency', async () => {
  const file = storeFile()
  const browser = await load(file, async () => void (await import('@/store/translucency')))
  browser.stop()

  const custom = JSON.stringify({ mode: 'clear', base: { intensity: 61 }, light: {}, dark: {} })
  becomeNative({ [TRANSLUCENCY]: custom })
  const native = await load(file, async () => void (await import('@/store/translucency')))
  native.stop()

  expect(window.localStorage.getItem(TRANSLUCENCY)).toBe(custom)
  expect(readUiPreferences(file).values[TRANSLUCENCY]).toBe(custom)
})

it('a fresh browser’s auto-speak migration never replaces native’s choice, across a browser reload too', async () => {
  const file = storeFile()

  const migrate = async () => {
    const prefs = await import('@/store/voice-prefs')
    prefs.applyAutoSpeakFromConfig({ voice: { auto_tts: false } })
  }

  ;(await load(file, migrate)).stop()
  // The reload finds its migrated value already in localStorage.
  ;(await load(file, migrate)).stop()
  expect(window.localStorage.getItem(AUTO_SPEAK)).toBe('false')

  becomeNative({ [AUTO_SPEAK]: 'true' })
  const native = await load(file, migrate)
  native.stop()

  expect(window.localStorage.getItem(AUTO_SPEAK)).toBe('true')
  expect(readUiPreferences(file).values[AUTO_SPEAK]).toBe('true')
})

it('a migrated auto-speak value stays put here while the store lacks it, and a real choice is shared', async () => {
  const file = storeFile()
  becomeNative({ [THEME]: 'nous' })
  const choose: { autoSpeak?: (enabled: boolean) => Promise<void> } = {}

  const native = await load(file, async () => {
    const prefs = await import('@/store/voice-prefs')
    prefs.applyAutoSpeakFromConfig({ voice: { auto_tts: true } })
    choose.autoSpeak = prefs.setAutoSpeakReplies
  })

  // Another client's change reaches this joined window: nothing it lacks is pruned.
  await writeUiPreferences(file, { values: { [THEME]: 'mono' } })
  native.push()
  await tick()
  expect(window.localStorage.getItem(AUTO_SPEAK)).toBe('true')
  expect(readUiPreferences(file).values).toEqual({ [THEME]: 'mono' })

  await choose.autoSpeak?.(false)
  await tick()
  native.stop()

  expect(readUiPreferences(file).values).toEqual({ [AUTO_SPEAK]: 'false', [THEME]: 'mono' })
})
