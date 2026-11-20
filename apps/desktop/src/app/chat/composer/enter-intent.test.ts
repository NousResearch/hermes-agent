import { describe, expect, it } from 'vitest'

import { resolveEnterIntent } from './enter-intent'

describe('resolveEnterIntent', () => {
  it('breaks the line on a touch host for a plain Return', () => {
    expect(resolveEnterIntent({ key: 'Enter', touch: true })).toBe('break')
  })

  // The regression @kvnloo flagged: a shifted Return on a touch host used to
  // skip the explicit line break and fall through to the browser.
  it('breaks the line on a touch host for a shifted Return too', () => {
    expect(resolveEnterIntent({ key: 'Enter', shiftKey: true, touch: true })).toBe('break')
  })

  it('never sends from a keyboard on a touch host', () => {
    expect(resolveEnterIntent({ key: 'Enter', touch: true })).not.toBe('send')
    expect(resolveEnterIntent({ key: 'Enter', shiftKey: true, touch: true })).not.toBe('send')
  })

  it('sends on a desktop Return and leaves Shift+Return to the browser', () => {
    expect(resolveEnterIntent({ key: 'Enter', touch: false })).toBe('send')
    expect(resolveEnterIntent({ key: 'Enter', shiftKey: true, touch: false })).toBe('native')
  })

  it('queues on Cmd/Ctrl+Enter on every host, including touch', () => {
    expect(resolveEnterIntent({ key: 'Enter', metaKey: true, touch: false })).toBe('queue')
    expect(resolveEnterIntent({ key: 'Enter', ctrlKey: true, touch: true })).toBe('queue')
  })

  it('leaves every other key alone', () => {
    for (const key of ['a', 'Escape', 'ArrowUp', 'Tab']) {
      expect(resolveEnterIntent({ key, touch: true })).toBe('native')
      expect(resolveEnterIntent({ key, touch: false })).toBe('native')
    }
  })
})
