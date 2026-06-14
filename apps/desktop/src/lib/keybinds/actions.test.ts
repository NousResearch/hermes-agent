import { describe, expect, it } from 'vitest'

import { defaultBindings, keybindAction, keybindActionAllowedInEditableTarget } from './actions'

describe('desktop keybind actions', () => {
  it('exposes unbound reasoning level actions for the keyboard shortcuts panel', () => {
    expect(keybindAction('composer.reasoningUp')).toMatchObject({
      category: 'composer',
      defaults: [],
      editableTargetPolicy: 'modified'
    })
    expect(keybindAction('composer.reasoningDown')).toMatchObject({
      category: 'composer',
      defaults: [],
      editableTargetPolicy: 'modified'
    })
    expect(defaultBindings()['composer.reasoningUp']).toEqual([])
    expect(defaultBindings()['composer.reasoningDown']).toEqual([])
  })

  it('allows reasoning level shortcuts to fire while the composer is focused', () => {
    expect(keybindActionAllowedInEditableTarget('composer.reasoningUp', 'alt+]')).toBe(true)
    expect(keybindActionAllowedInEditableTarget('composer.reasoningDown', 'mod+alt+[')).toBe(true)
    expect(keybindActionAllowedInEditableTarget('composer.reasoningUp', ']')).toBe(false)
    expect(keybindActionAllowedInEditableTarget('composer.reasoningUp', 'shift+]')).toBe(false)
    expect(keybindActionAllowedInEditableTarget('appearance.toggleMode', 'alt+x')).toBe(false)
  })
})
