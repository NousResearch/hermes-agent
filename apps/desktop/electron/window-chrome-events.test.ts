import { EventEmitter } from 'node:events'

import { describe, expect, it, vi } from 'vitest'

import { bindWindowChromeEvents } from './window-chrome-events'

describe('window chrome events', () => {
  it.each(['maximize', 'unmaximize', 'minimize', 'restore', 'hide', 'show'])(
    'publishes %s for the emitting window only',
    event => {
      const primary = new EventEmitter()
      const secondary = new EventEmitter()
      const publish = vi.fn()
      bindWindowChromeEvents(primary, publish)
      bindWindowChromeEvents(secondary, publish)

      secondary.emit(event)

      expect(publish).toHaveBeenCalledExactlyOnceWith(undefined, secondary)
    }
  )

  it.each([
    ['will-enter-full-screen', true],
    ['enter-full-screen', true],
    ['will-leave-full-screen', false],
    ['leave-full-screen', false]
  ] as const)('routes %s to its owner with the transition state', (event, fullscreen) => {
    const primary = new EventEmitter()
    const secondary = new EventEmitter()
    const publish = vi.fn()
    bindWindowChromeEvents(primary, publish)
    bindWindowChromeEvents(secondary, publish)

    secondary.emit(event)

    expect(publish).toHaveBeenCalledExactlyOnceWith(fullscreen, secondary)
  })
})
