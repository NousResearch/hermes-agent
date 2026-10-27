import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, it } from 'vitest'

import { parseUiPreferencesWrite, readUiPreferences, writeUiPreferences } from '../../../electron/ui-preferences'

import { bootRenderer } from './boot'
import { startUiPreferencesSync, type UiPreferencesBridge } from './sync'

const FAVORITES = 'hermes.desktop.favorite-models'
const dirs: string[] = []
const tick = (ms = 400) => new Promise(resolve => setTimeout(resolve, ms))

afterEach(() => {
  window.localStorage.clear()
  delete window.hermesDesktop

  for (const dir of dirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

// The production chunk graph evaluates every shared chunk (where most stores
// live) before the entry body, so hydration must gate the app's module graph
// itself: a store that read localStorage first kept a stale value, and its
// next write replaced the host's favourites (#145536 review, finding 1).
it('no store module evaluates before the machine store hydrated, so its next write keeps the host’s values', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-ui-boot-'))
  dirs.push(dir)
  const file = path.join(dir, 'ui-preferences.json')
  await writeUiPreferences(file, { values: { [FAVORITES]: '["host::A","host::B"]' } })

  const bridge: UiPreferencesBridge = {
    snapshot: readUiPreferences(file),
    read: async () => readUiPreferences(file),
    write: payload => writeUiPreferences(file, parseUiPreferencesWrite(payload))
  }

  window.hermesDesktop = { uiPreferences: bridge } as unknown as typeof window.hermesDesktop

  const favorites = await bootRenderer(() => import('@/store/favorite-models'))
  const stop = startUiPreferencesSync(bridge)
  await tick()
  favorites.toggleFavoriteModel('mine', 'C')
  await tick()
  stop()

  expect(JSON.parse(readUiPreferences(file).values[FAVORITES])).toEqual(['host::A', 'host::B', 'mine::C'])
})
