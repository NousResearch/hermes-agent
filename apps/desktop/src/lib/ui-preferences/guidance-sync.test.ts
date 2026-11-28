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
    firstTimeSplash: await import('@/store/first-time-splash'),
    onboarding: await import('@/store/onboarding'),
    renameSplash: await import('@/store/rename-splash'),
    starterIdeas: await import('@/store/starter-ideas'),
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

it('the rename splash seen in one browser is not shown again in another', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-ui-guidance-'))
  dirs.push(dir)
  const file = path.join(dir, 'ui-preferences.json')

  const first = await load(file)
  await tick()
  first.renameSplash.considerRenameSplash('existing')
  first.renameSplash.dismissRenameSplash('button')
  await tick()
  first.stop()

  const second = await load(file)
  await tick()
  second.renameSplash.considerRenameSplash('existing')
  second.stop()

  expect(second.renameSplash.$renameSplash.get()).toBeNull()
  expect(second.renameSplash.$renameSplashReopenable.get()).toBe(true)
})

it('a newcomer’s first-time splash and finished tasks carry to their other browsers', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-ui-guidance-'))
  dirs.push(dir)
  const file = path.join(dir, 'ui-preferences.json')

  const first = await load(file)
  await tick()
  first.renameSplash.considerRenameSplash('fresh')
  first.firstTimeSplash.considerFirstTimeSplash('fresh')
  first.firstTimeSplash.closeFirstTimeSplash('close')
  first.starterIdeas.noteTaskFinished('chat-a')
  first.starterIdeas.noteTaskFinished('chat-b')
  await tick()
  first.stop()

  // A phone: its own empty storage, and by now the machine has chats, so its verdict is `existing`.
  const second = await load(file)
  await tick()
  second.renameSplash.considerRenameSplash('existing')
  second.firstTimeSplash.considerFirstTimeSplash('existing')
  second.stop()

  expect(second.firstTimeSplash.$firstTimeSplash.get()).toBeNull()
  expect(second.firstTimeSplash.$firstTimeSplashReopenable.get()).toBe(true)
  expect(second.renameSplash.$renameSplash.get()).toBeNull()
  expect(second.starterIdeas.$starterIdeas.get()).toBe('retired')
})
