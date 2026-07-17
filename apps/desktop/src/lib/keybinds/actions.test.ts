import { describe, expect, it } from 'vitest'

import { keybindAction } from './actions'

describe('composer dictation keybind', () => {
  it('ships on Ctrl+Shift+D off macOS through the cross-platform mod chord', () => {
    expect(keybindAction('composer.dictation')?.defaults).toEqual(['mod+shift+d'])
  })
})
