import { describe, expect, it } from 'vitest'

import { browserTabShortcut } from './browser-tab-shortcuts'

const input = { type: 'keyDown', key: 't', code: 'KeyT', control: true, meta: false, alt: false, shift: false }

const bindings = {
  'session.newTab': ['mod+t'],
  'view.closeTab': ['mod+w'],
  'session.next': ['ctrl+tab'],
  'session.prev': ['ctrl+shift+tab']
}

describe('scoped detached-browser chords', () => {
  it('uses Control on Windows/Linux and Command on macOS', () => {
    expect(browserTabShortcut(input, bindings, false)).toBe('session.newTab')
    expect(browserTabShortcut(input, bindings, true)).toBeNull()
    expect(browserTabShortcut({ ...input, control: false, meta: true }, bindings, true)).toBe('session.newTab')
  })
  it('cycles with physical Control on all platforms', () => {
    for (const mac of [false, true]) {
      expect(browserTabShortcut({ ...input, key: 'Tab', code: 'Tab' }, bindings, mac)).toBe('session.next')
      expect(browserTabShortcut({ ...input, key: 'Tab', code: 'Tab', shift: true }, bindings, mac)).toBe('session.prev')
    }
  })
  it('respects remapped and cleared chords rather than shipping an unconfigurable accelerator', () => {
    expect(browserTabShortcut(input, {}, false)).toBeNull()
    const custom = { 'session.newTab': ['mod+shift+]'] }
    expect(browserTabShortcut({ ...input, key: '}', code: 'BracketRight', shift: true }, custom, false)).toBe(
      'session.newTab'
    )
    expect(browserTabShortcut(input, custom, false)).toBeNull()
  })
  it('leaves typing, composing, synthetic modifier keys, key-up and repeat alone', () => {
    for (const variant of [
      { control: false },
      { isComposing: true },
      { key: 'Process' },
      { key: 'Control' },
      { type: 'keyUp' },
      { isAutoRepeat: true }
    ]) {
      expect(browserTabShortcut({ ...input, ...variant }, bindings, false)).toBeNull()
    }

    expect(
      browserTabShortcut({ ...input, control: false, shift: true }, { 'session.newTab': ['shift+t'] }, false)
    ).toBeNull()
  })
})
