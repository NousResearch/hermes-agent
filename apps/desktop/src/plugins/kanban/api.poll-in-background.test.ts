import { afterEach, describe, expect, it } from 'vitest'

import { isBrowserHosted, pollInBackground } from './api'

// A hidden window must not keep a Desktop-started local gateway awake: its idle-exit never fires
// while something keeps asking it for the board. Everything else keeps the fallback poll.

describe('pollInBackground', () => {
  it('never polls a hidden window while the event channel is live', () => {
    for (const scope of ['local', 'work']) {
      for (const browserHosted of [false, true]) {
        expect(pollInBackground('live', scope, browserHosted)).toBe(false)
      }
    }
  })

  it('keeps a local Desktop gateway idle-able: no background poll', () => {
    expect(pollInBackground('polling', 'local', false)).toBe(false)
    expect(pollInBackground('offline', 'local', false)).toBe(false)
  })

  it('polls in the background for a remote connection', () => {
    expect(pollInBackground('polling', 'work', false)).toBe(true)
    expect(pollInBackground('offline', 'work', false)).toBe(true)
  })

  it('polls in the background in the browser-hosted Webapp, whose scope reads as local', () => {
    expect(pollInBackground('polling', 'local', true)).toBe(true)
    expect(pollInBackground('offline', 'local', true)).toBe(true)
  })
})

describe('isBrowserHosted', () => {
  afterEach(() => {
    delete document.documentElement.dataset.hermesDesktopHost
  })

  it('reads the host marker the browser bridge writes', () => {
    expect(isBrowserHosted()).toBe(false)
    document.documentElement.dataset.hermesDesktopHost = 'browser'
    expect(isBrowserHosted()).toBe(true)
  })
})
