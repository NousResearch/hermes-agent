import { beforeEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'hermes.desktop.keybinds'

function storedDiff(): Record<string, string[]> {
  return JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? '{}')
}

describe('keybinds store boot persist', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('keeps stored overrides for plugin actions that register after boot', async () => {
    // A plugin-action rebind saved by an earlier session. The plugin has not
    // registered yet at boot, so the id is unknown to `allKeybindActions()`.
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ 'demo.late': ['mod+alt+l'] }))

    await import('./keybinds')

    // The boot-time subscribe persist must carry the unknown id forward
    // instead of overwriting storage with only the registered actions.
    expect(storedDiff()).toEqual({ 'demo.late': ['mod+alt+l'] })
  })

  it('keeps unregistered overrides when the user rebinds a built-in', async () => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ 'demo.late': ['mod+alt+l'] }))

    const { setBinding } = await import('./keybinds')
    setBinding('session.new', ['mod+shift+n'])

    expect(storedDiff()).toEqual({
      'demo.late': ['mod+alt+l'],
      'session.new': ['mod+shift+n'],
    })
  })
})
