import { afterEach, describe, expect, it } from 'vitest'

// The bindings store every keybind surface reads; it declares window-dependent availability itself.
import { $bindings, conflictsFor } from '@/store/keybinds'
import { $shellChoice } from '@/store/shell-variant'

import { $offeredKeybindActions, allKeybindActions, defaultBindings } from './actions'

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

  it('updates a live list of offered actions when the shell flips', () => {
    const seen: boolean[] = []

    // Other shell-dependent rows (Back/Forward) re-emit the list on the same flip; count changes only.
    const stop = $offeredKeybindActions.subscribe(actions => {
      const listedNow = actions.some(action => action.id === 'view.toggleStatusbar')

      if (seen.at(-1) !== listedNow) {
        seen.push(listedNow)
      }
    })

    $shellChoice.set(false)
    $shellChoice.set(null)
    stop()

    expect(seen).toEqual([false, true, false])
  })

  it('keeps its binding while unlisted, so a flip back to classic finds the chord where it was', () => {
    $shellChoice.set(null)

    expect(defaultBindings()['view.toggleStatusbar']).toEqual(['mod+shift+s'])
    expect($bindings.get()['view.toggleStatusbar']).toEqual(['mod+shift+s'])
    expect(conflictsFor('view.toggleTabStrip', 'mod+shift+s')).toContain('view.toggleStatusbar')
  })
})
