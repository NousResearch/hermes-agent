import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import type { Theme } from '@tauri-apps/api/window'

import { resolveTheme, watchTheme } from './theme.ts'
import type { ThemeTrackerDeps } from './theme.ts'

/*
 * Regression tests for the installer OS-appearance follower.
 *
 * The macOS failure mode: the one-shot Tauri read is trusted absolutely, and
 * when it resolves null (or stale) the page keeps only its first paint with
 * no live fallback — the installer sits in light mode on a dark system
 * forever. These tests pin the invariants: an explicit window theme wins,
 * null falls back to the media query AND stays live, and every signal
 * re-reads the backend so a stale startup read heals.
 *
 * Runs on the Node built-in runner with no dependencies:
 *   node --experimental-strip-types --test src/theme.test.ts
 * (theme.ts imports the Tauri API type-only, so nothing resolves it here.)
 */

interface FakeState {
  tauriTheme: Theme | null
  readFails: boolean
  subscribeFails: boolean
  mediaDark: boolean
  applied: Theme[]
  readCalls: number
  mediaCbs: Array<() => void>
  tauriCbs: Array<(theme: Theme) => void>
  settleCbs: Array<() => void>
}

function makeState(overrides: Partial<FakeState> = {}): FakeState {
  return {
    tauriTheme: null,
    readFails: false,
    subscribeFails: false,
    mediaDark: false,
    applied: [],
    readCalls: 0,
    mediaCbs: [],
    tauriCbs: [],
    settleCbs: [],
    ...overrides
  }
}

function makeDeps(state: FakeState): ThemeTrackerDeps {
  return {
    readWindowTheme: () => {
      state.readCalls += 1

      if (state.readFails) {
        return Promise.reject(new Error('no backend'))
      }

      return Promise.resolve(state.tauriTheme)
    },
    onWindowThemeChanged: cb => {
      if (state.subscribeFails) {
        return Promise.reject(new Error('no backend events'))
      }

      state.tauriCbs.push(cb)

      return Promise.resolve()
    },
    mediaDark: () => state.mediaDark,
    onMediaDarkChanged: cb => {
      state.mediaCbs.push(cb)
    },
    applyTheme: theme => {
      state.applied.push(theme)
    },
    afterFirstFrames: cb => {
      state.settleCbs.push(cb)
    }
  }
}

const tick = (): Promise<void> => new Promise(resolve => setImmediate(resolve))

void describe('resolveTheme', () => {
  void it('prefers an explicit window theme over the media query', () => {
    assert.equal(resolveTheme('dark', false), 'dark')
    assert.equal(resolveTheme('light', true), 'light')
  })

  void it('falls back to the media query when the window theme is unknown', () => {
    assert.equal(resolveTheme(null, true), 'dark')
    assert.equal(resolveTheme(null, false), 'light')
  })
})

void describe('watchTheme', () => {
  void it('paints the explicit window theme on startup', async () => {
    // WebView2/WebKitGTK media queries are unreliable: Tauri wins.
    const state = makeState({ tauriTheme: 'dark', mediaDark: false })
    await watchTheme(makeDeps(state))

    assert.deepEqual(state.applied, ['dark'])
  })

  void it('tracks the media query live when the window theme is unknown', async () => {
    const state = makeState({ tauriTheme: null, mediaDark: true })
    await watchTheme(makeDeps(state))
    assert.deepEqual(state.applied, ['dark'])

    // The OS flips to light later: with no backend answer, media must drive.
    state.mediaDark = false

    for (const cb of state.mediaCbs) { cb() }
    await tick()

    assert.deepEqual(state.applied, ['dark', 'light'])
  })

  void it('repaints on backend theme-changed events', async () => {
    const state = makeState({ tauriTheme: 'light', mediaDark: false })
    await watchTheme(makeDeps(state))
    assert.deepEqual(state.applied, ['light'])

    state.tauriTheme = 'dark'

    for (const cb of state.tauriCbs) { cb('dark') }
    await tick()

    assert.deepEqual(state.applied, ['light', 'dark'])
  })

  void it('re-reads the backend on media changes and after first frames', async () => {
    // Startup race: the one-shot read says light while the settled answer
    // is dark. The settle re-read must heal it.
    const state = makeState({ tauriTheme: 'light', mediaDark: true })
    await watchTheme(makeDeps(state))
    assert.deepEqual(state.applied, ['light'])

    state.tauriTheme = 'dark'

    for (const cb of state.settleCbs) { cb() }
    await tick()
    assert.deepEqual(state.applied, ['light', 'dark'])

    // A later media flip re-reads the backend instead of trusting the
    // (possibly stale) media value alone.
    const reads = state.readCalls
    state.tauriTheme = 'light'
    state.mediaDark = false

    for (const cb of state.mediaCbs) { cb() }
    await tick()

    assert.ok(state.readCalls > reads)
    assert.deepEqual(state.applied.slice(-1), ['light'])
  })

  void it('degrades to the media query when the backend is unreachable', async () => {
    const state = makeState({ readFails: true, subscribeFails: true, mediaDark: true })
    await watchTheme(makeDeps(state))
    assert.deepEqual(state.applied, ['dark'])

    state.mediaDark = false

    for (const cb of state.mediaCbs) { cb() }
    await tick()

    assert.deepEqual(state.applied, ['dark', 'light'])
  })

  void it('keeps a working explicit theme across transient backend failures', async () => {
    const state = makeState({ tauriTheme: 'dark', mediaDark: false })
    await watchTheme(makeDeps(state))
    assert.deepEqual(state.applied, ['dark'])

    // The backend answered, then starts failing while the OS toggles:
    // the latched explicit theme must survive, not fall back to media.
    state.readFails = true
    state.mediaDark = true

    for (const cb of state.mediaCbs) { cb() }
    await tick()

    assert.deepEqual(state.applied, ['dark', 'dark'])
  })
})
