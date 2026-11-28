import { afterEach, describe, expect, it } from 'vitest'

// The bindings store every keybind surface reads; it declares window-dependent availability itself.
import { $bindings, $comboIndex, bindingsFor, conflictsFor } from '@/store/keybinds'
import { $shellChoice } from '@/store/shell-variant'

import { $offeredKeybindActions, allKeybindActions, defaultBindings, keybindActionAvailable } from './actions'
import { canonicalizeCombo } from './combo'

const listed = (id: string) => allKeybindActions().some(action => action.id === id)

afterEach(() => $shellChoice.set(null))

// Both shells have a status bar (classic's strip and the redesign's bar), so ⌘⇧S stays available;
// the profile rail is classic-only and tracks shell flips.
describe('shell-dependent shortcuts', () => {
  it('lists the statusbar toggle in both classic and redesign shells', () => {
    $shellChoice.set(false)
    expect(listed('view.toggleStatusbar')).toBe(true)

    $shellChoice.set(true)
    expect(listed('view.toggleStatusbar')).toBe(true)
  })

  it('lists the profile rail toggle only in the shell that has a profile rail', () => {
    $shellChoice.set(false)
    expect(listed('view.toggleProfileRail')).toBe(true)

    $shellChoice.set(true)
    expect(listed('view.toggleProfileRail')).toBe(false)
  })

  it('updates a live list of offered actions when the shell flips', () => {
    const seen: boolean[] = []

    // Other shell-dependent rows (Back/Forward) re-emit the list on the same flip; count changes only.
    const stop = $offeredKeybindActions.subscribe(actions => {
      const listedNow = actions.some(action => action.id === 'view.toggleProfileRail')

      if (seen.at(-1) !== listedNow) {
        seen.push(listedNow)
      }
    })

    $shellChoice.set(false)
    $shellChoice.set(null)
    stop()

    expect(seen).toEqual([false, true, false])
  })

  it('keeps the statusbar default chord in both shells', () => {
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
