import { describe, expect, it } from 'vitest'

import { $comboIndex } from '@/store/keybinds'

import { defaultBindings, KEYBIND_ACTIONS, keybindAction } from './actions'
import { canonicalizeCombo } from './combo'

// The split-pane toggle keybind (design §6 step 14): registered in the action
// registry, shipped on ⌘⇧\ — and cleanly distinct from ⌘\ (view.flipPanes).

describe('view.toggleSplit keybind', () => {
  it('is registered in the rebindable registry under the view category', () => {
    const action = keybindAction('view.toggleSplit')

    expect(action).toBeDefined()
    expect(action?.category).toBe('view')
    expect(action?.defaults).toEqual(['mod+shift+\\'])
  })

  it('ships its default combo through defaultBindings()', () => {
    expect(defaultBindings()['view.toggleSplit']).toEqual(['mod+shift+\\'])
  })

  it('resolves through the live combo index without clashing with view.flipPanes', () => {
    const index = $comboIndex.get()

    expect(index.get(canonicalizeCombo('mod+shift+\\'))).toBe('view.toggleSplit')
    expect(index.get(canonicalizeCombo('mod+\\'))).toBe('view.flipPanes')
  })

  it('collides with no other shipped default', () => {
    const combo = canonicalizeCombo('mod+shift+\\')

    const owners = KEYBIND_ACTIONS.filter(action =>
      action.defaults.some(candidate => canonicalizeCombo(candidate) === combo)
    ).map(action => action.id)

    expect(owners).toEqual(['view.toggleSplit'])
  })
})
