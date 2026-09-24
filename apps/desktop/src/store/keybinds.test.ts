import { beforeEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'hermes.desktop.keybinds'

function storedDiff(): Record<string, string[]> {
  return JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? '{}')
}

const DEMO_CONTRIBUTION = {
  data: { id: 'demo.late', label: 'Demo', run: () => undefined },
  id: 'demo:late',
  plugin: 'demo'
}

// #116331: the writer (`persistBindings`) diffs over the action universe at
// call time, while the reader keeps unknown ids verbatim for plugins that
// register late. A persist that runs while a contributed action is not (yet /
// any more) registered must not wipe its stored override.
describe('keybinds store persist vs late-registered contributed actions', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('keeps stored overrides for plugin actions that register after boot', async () => {
    // A plugin-action rebind saved by an earlier session. The plugin has not
    // registered yet at boot, so the id is unknown to `allKeybindActions()`.
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ 'demo.late': ['mod+alt+l'] }))

    const { bindingsFor } = await import('./keybinds')

    // The boot-time subscribe persist must carry the unknown id forward
    // instead of overwriting storage with only the registered actions.
    expect(storedDiff()).toEqual({ 'demo.late': ['mod+alt+l'] })

    // Once the plugin registers, the saved rebind resolves for its action.
    const { registry } = await import('@/contrib/registry')
    const { KEYBINDS_AREA } = await import('@/lib/keybinds/actions')
    registry.register({ area: KEYBINDS_AREA, ...DEMO_CONTRIBUTION })
    expect(bindingsFor('demo.late')).toEqual(['mod+alt+l'])
  })

  it('keeps an override written after boot when the plugin unloads and another action is rebound', async () => {
    const { bindingsFor, setBinding } = await import('./keybinds')
    const { registry } = await import('@/contrib/registry')
    const { KEYBINDS_AREA } = await import('@/lib/keybinds/actions')

    const unload = registry.register({ area: KEYBINDS_AREA, ...DEMO_CONTRIBUTION })
    setBinding('demo.late', ['mod+alt+l'])
    expect(storedDiff()).toEqual({ 'demo.late': ['mod+alt+l'] })

    unload()
    setBinding('session.new', ['mod+shift+n'])

    expect(storedDiff()).toEqual({
      'demo.late': ['mod+alt+l'],
      'session.new': ['mod+shift+n']
    })
    expect(bindingsFor('demo.late')).toEqual(['mod+alt+l'])
  })
})
