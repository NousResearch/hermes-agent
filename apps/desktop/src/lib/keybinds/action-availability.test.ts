// Declares the statusbar toggle's availability, as the app does when it loads the statusbar.
import '@/store/statusbar-prefs'

import { afterEach, describe, expect, it } from 'vitest'

import { $bindings, conflictsFor } from '@/store/keybinds'
import { $shellChoice } from '@/store/shell-variant'

import { allKeybindActions, defaultBindings } from './actions'

const listed = (id: string) => allKeybindActions().some(action => action.id === id)

afterEach(() => $shellChoice.set(null))

// ⌘⇧S toggles the classic statusbar; the redesign has no strip, so there the row would do nothing.
describe('shell-dependent shortcuts', () => {
  it('lists the statusbar toggle only in the shell that has a statusbar', () => {
    $shellChoice.set(false)
    expect(listed('view.toggleStatusbar')).toBe(true)

    $shellChoice.set(null)
    expect(listed('view.toggleStatusbar')).toBe(false)
  })

  it('keeps its binding while unlisted, so a flip back to classic finds the chord where it was', () => {
    $shellChoice.set(null)

    expect(defaultBindings()['view.toggleStatusbar']).toEqual(['mod+shift+s'])
    expect($bindings.get()['view.toggleStatusbar']).toEqual(['mod+shift+s'])
    expect(conflictsFor('view.toggleTabStrip', 'mod+shift+s')).toContain('view.toggleStatusbar')
  })
})
