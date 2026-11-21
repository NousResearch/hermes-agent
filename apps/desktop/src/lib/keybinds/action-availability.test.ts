import { afterEach, describe, expect, it } from 'vitest'

// The bindings store every keybind surface reads; it declares window-dependent availability itself.
import { $bindings, $comboIndex, bindingsFor, conflictsFor } from '@/store/keybinds'
import { $shellChoice } from '@/store/shell-variant'

import { $offeredKeybindActions, allKeybindActions, defaultBindings, keybindActionAvailable } from './actions'
import { canonicalizeCombo } from './combo'

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

// "New agent" is the redesign's Agents-page button. Classic (main) has no such shortcut: the
// shortcut map does not list it and the new-chat chord goes straight to New chat.
describe('the New agent shortcut', () => {
  const chordTakers = () =>
    bindingsFor('agents.new').map(combo =>
      ($comboIndex.get().get(canonicalizeCombo(combo)) ?? []).filter(keybindActionAvailable)
    )

  it('is neither listed nor dispatched in classic', () => {
    $shellChoice.set(false)

    expect(listed('agents.new')).toBe(false)
    expect(listed('session.new')).toBe(true)
    expect(chordTakers().every(ids => ids.length === 1 && ids[0] === 'session.new')).toBe(true)
  })

  it('is listed and takes the chord first in the redesign', () => {
    $shellChoice.set(true)

    expect(listed('agents.new')).toBe(true)
    expect(chordTakers().every(ids => ids[0] === 'agents.new' && ids.includes('session.new'))).toBe(true)
  })
})
