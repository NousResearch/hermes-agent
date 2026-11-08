import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, it, vi } from 'vitest'

import { parseUiPreferencesWrite, readUiPreferences, writeUiPreferences } from '../../../electron/ui-preferences'

const dirs: string[] = []
const tick = (ms = 400) => new Promise(resolve => setTimeout(resolve, ms))

afterEach(() => {
  window.localStorage.clear()
  window.sessionStorage.clear()

  for (const dir of dirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

/** One page load of an origin against the machine store file, as boot.ts + main.tsx run it. */
async function load(file: string) {
  window.localStorage.clear()
  window.sessionStorage.clear()
  vi.resetModules()
  const sync = await import('./sync')
  sync.captureOwnPreferences()
  sync.hydrateFromSnapshot(readUiPreferences(file))

  const stop = sync.startUiPreferencesSync({
    read: async () => readUiPreferences(file),
    write: payload => writeUiPreferences(file, parseUiPreferencesWrite(payload))
  })

  return {
    onboarding: await import('@/store/onboarding'),
    stop,
    tips: await import('@/store/tips')
  }
}

it('onboarding skipped and tips seen or dismissed in one browser stay so in another (#145600)', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-ui-guidance-'))
  dirs.push(dir)
  const file = path.join(dir, 'ui-preferences.json')

  const first = await load(file)
  await tick()
  first.onboarding.dismissFirstRunOnboarding()
  first.tips.showTip({ side: 'top', targets: [], text: 'One', tipId: 'catalog:one' })
  first.tips.retireTips(['catalog:two'])
  await tick()
  first.stop()

  // A phone, or the same Webapp on another origin: its own empty localStorage.
  const second = await load(file)
  await tick()
  second.stop()

  expect(second.onboarding.$desktopOnboarding.get().firstRunSkipped).toBe(true)
  expect(second.tips.$retiredTips.get()).toContain('catalog:two')
  expect(Object.keys(second.tips.$tipShownAt.get())).toContain('catalog:one')
})
