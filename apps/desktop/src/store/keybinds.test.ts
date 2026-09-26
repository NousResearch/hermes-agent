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

  it('keeps an explicitly cleared sidebar binding empty so mod+b can be unbound', async () => {
    const { $comboIndex, bindingsFor, setBinding } = await import('./keybinds')

    setBinding('view.toggleSidebar', [])

    expect(bindingsFor('view.toggleSidebar')).toEqual([])
    expect($comboIndex.get().get('mod+b')).toBeUndefined()
    expect(storedDiff()['view.toggleSidebar']).toEqual([])

    vi.resetModules()
    const reloaded = await import('./keybinds')

    expect(reloaded.bindingsFor('view.toggleSidebar')).toEqual([])
    expect(reloaded.$comboIndex.get().get('mod+b')).toBeUndefined()
  })

  it('treats Backspace and Delete during capture as a cleared binding', async () => {
    const keybinds = await import('./keybinds')

    expect(keybinds.captureStep('Backspace', null)).toEqual({ type: 'set', combos: [] })
    expect(keybinds.captureStep('Delete', 'mod+b')).toEqual({ type: 'set', combos: [] })
    expect(keybinds.captureStep('Escape', null)).toEqual({ type: 'cancel' })
    expect(keybinds.captureStep('b', null)).toEqual({ type: 'wait' })
    expect(keybinds.captureStep('b', 'mod+b')).toEqual({ type: 'set', combos: ['mod+b'] })
  })
})
