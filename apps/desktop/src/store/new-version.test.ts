import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { entryModuleOf, isNewerBuild, mayApplyOnNewChat } from '@/lib/deployed-build'

import { $newVersionReady, applyNewVersionOnNewChat, checkForNewVersion, watchForNewVersion } from './new-version'
import { $voiceStreamPaintOwners } from './voice-stream-paint'

const index = (entry: string) =>
  `<!doctype html><html><head><script>window.__HERMES_BASE_PATH__="";</script>` +
  `<script type="module" crossorigin src="${entry}"></script>` +
  `<link rel="stylesheet" crossorigin href="./assets/index-aaa.css"></head><body></body></html>`

beforeEach(() => {
  $newVersionReady.set(false)
  $voiceStreamPaintOwners.set(0)
})

afterEach(() => vi.useRealTimers())

describe('telling a newer build apart', () => {
  it('reads the entry module a served index loads, whatever the host injected around it', () => {
    expect(entryModuleOf(index('./assets/index-B1.js'))).toBe('./assets/index-B1.js')
    expect(entryModuleOf('<script src="./legacy.js"></script>')).toBeNull()
    expect(entryModuleOf('<html><body>Hermes is unavailable</body></html>')).toBeNull()
  })

  it('counts only a different entry in the same place as a new build', () => {
    expect(isNewerBuild('./assets/index-A.js', './assets/index-B.js')).toBe(true)
    expect(isNewerBuild('./assets/index-A.js', './assets/index-A.js')).toBe(false)
    expect(isNewerBuild('./assets/index-A.js', 'https://sign-in.example/app.js')).toBe(false)
    expect(isNewerBuild('./assets/index-A.js', null)).toBe(false)
    expect(isNewerBuild(null, './assets/index-B.js')).toBe(false)
  })

  it('marks the version ready from the probe, and an unreadable index changes nothing', async () => {
    await checkForNewVersion({ running: './assets/index-A.js', served: async () => null })
    expect($newVersionReady.get()).toBe(false)

    await checkForNewVersion({ running: './assets/index-A.js', served: async () => './assets/index-B.js' })
    expect($newVersionReady.get()).toBe(true)
  })

  it('checks only while the tab is visible, and not again on every return to it', async () => {
    vi.useFakeTimers()
    const served = vi.fn(async () => './assets/index-A.js')
    let visibility: DocumentVisibilityState = 'hidden'
    vi.spyOn(document, 'visibilityState', 'get').mockImplementation(() => visibility)

    const stop = watchForNewVersion({ running: './assets/index-A.js', served })
    expect(served).not.toHaveBeenCalled()

    visibility = 'visible'
    document.dispatchEvent(new Event('visibilitychange'))
    document.dispatchEvent(new Event('visibilitychange'))
    expect(served).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(10 * 60_000)
    expect(served).toHaveBeenCalledTimes(2)
    stop()
  })
})

describe('when the new build may load', () => {
  it('waits for running turns and spoken replies before a new chat applies it', () => {
    expect(mayApplyOnNewChat({ workingChats: 0, voiceActive: false })).toBe(true)
    expect(mayApplyOnNewChat({ workingChats: 1, voiceActive: false })).toBe(false)
    expect(mayApplyOnNewChat({ workingChats: 0, voiceActive: true })).toBe(false)
  })

  it('reloads on a new chat only once a new build is ready and nothing would be cut short', () => {
    const reload = vi.fn()

    expect(applyNewVersionOnNewChat(reload)).toBe(false)

    $newVersionReady.set(true)
    $voiceStreamPaintOwners.set(1)
    expect(applyNewVersionOnNewChat(reload)).toBe(false)
    expect(reload).not.toHaveBeenCalled()

    $voiceStreamPaintOwners.set(0)
    expect(applyNewVersionOnNewChat(reload)).toBe(true)
    expect(reload).toHaveBeenCalledTimes(1)
  })
})
