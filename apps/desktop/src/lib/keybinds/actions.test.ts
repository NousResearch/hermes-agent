import { describe, expect, it } from 'vitest'

import { defaultBindings, keybindAction } from './actions'

describe('keybind actions', () => {
  it('ships a rebindable action for opening the integrated browser', () => {
    expect(keybindAction('view.openBrowser')).toMatchObject({ category: 'view' })
    expect(defaultBindings()['view.openBrowser']).toEqual(['mod+alt+b'])
  })
})
