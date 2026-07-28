import { describe, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { createPluginContext } from '@/contrib/plugin'
import { allKeybindActions, contributedKeybindHandler, KEYBINDS_AREA } from '@/lib/keybinds/actions'
import { bindingsFor } from '@/store/keybinds'

// The plugin-command contract: a plugin ships a hotkey through the `keybinds`
// area and it behaves like a built-in — it dispatches, it resolves a combo for
// the palette hint, and it survives the plugin being unloaded. These assert the
// relationship between the pieces, not the specific chord any one plugin picks.
describe('contributed keybind actions', () => {
  it('dispatches, resolves its default combo, and disappears on unload', () => {
    const ctx = createPluginContext('demo')
    let ran = 0

    const dispose = ctx.register({
      id: 'new-thing',
      area: KEYBINDS_AREA,
      data: {
        id: 'demo.newThing',
        category: 'view',
        defaults: ['mod+alt+n'],
        label: 'Demo: New thing',
        run: () => void (ran += 1)
      }
    })

    // Dispatch path: use-keybinds looks the handler up by action id.
    contributedKeybindHandler('demo.newThing')?.()
    expect(ran).toBe(1)

    // Hint path: $bindings was seeded before this action existed, so only the
    // resolver (default fallback) finds the combo — a raw store lookup can't.
    expect(bindingsFor('demo.newThing')).toEqual(['mod+alt+n'])

    // Panel path: it shows up as a rebindable row alongside the built-ins.
    expect(allKeybindActions().find(a => a.id === 'demo.newThing')?.label).toBe('Demo: New thing')

    dispose()

    expect(contributedKeybindHandler('demo.newThing')).toBeUndefined()
    expect(allKeybindActions().some(a => a.id === 'demo.newThing')).toBe(false)
  })

  it('cannot shadow a built-in action id', () => {
    const ctx = createPluginContext('demo')

    const dispose = ctx.register({
      id: 'steal-new-session',
      area: KEYBINDS_AREA,
      data: { id: 'session.new', defaults: ['mod+alt+n'], label: 'Demo: hijack', run: () => undefined }
    })

    // The built-in keeps its own combo and its own (i18n) label — the
    // contribution is filtered out rather than overriding core.
    expect(bindingsFor('session.new')).toEqual(['mod+n', 'shift+n'])
    expect(allKeybindActions().filter(a => a.id === 'session.new')).toHaveLength(1)

    dispose()
  })

  it('leaves no registry residue between plugin loads', () => {
    expect(registry.getArea(KEYBINDS_AREA).filter(c => c.source === 'plugin:demo')).toHaveLength(0)
  })
})
