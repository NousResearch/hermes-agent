import { describe, expect, it } from 'vitest'

import { applyPickerFilterKey, isPickerFilterKey, isPickerSearchOpen } from './picker-typeahead'

describe('picker-typeahead', () => {
  it('treats printable keys as filter keys', () => {
    expect(isPickerFilterKey({ altKey: false, ctrlKey: false, key: 'l', metaKey: false })).toBe(true)
    expect(isPickerFilterKey({ altKey: false, ctrlKey: false, key: 'Enter', metaKey: false })).toBe(false)
    expect(isPickerFilterKey({ altKey: false, ctrlKey: true, key: 'l', metaKey: false })).toBe(false)
  })

  it('edits the query from keystrokes', () => {
    expect(applyPickerFilterKey('', 'l')).toBe('l')
    expect(applyPickerFilterKey('lun', 'a')).toBe('luna')
    expect(applyPickerFilterKey('luna', 'Backspace')).toBe('lun')
    expect(applyPickerFilterKey('luna', 'Delete')).toBe('')
  })

  it('detects an open picker search field', () => {
    const host = document.createElement('div')
    host.innerHTML = '<div data-slot="select-search"><input /></div>'
    document.body.appendChild(host)

    expect(isPickerSearchOpen()).toBe(true)

    host.remove()
    expect(isPickerSearchOpen()).toBe(false)
  })
})
