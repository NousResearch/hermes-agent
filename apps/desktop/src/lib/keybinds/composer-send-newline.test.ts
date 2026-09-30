import { describe, expect, it } from 'vitest'

import { defaultBindings, KEYBIND_ACTIONS, KEYBIND_READONLY, keybindAction } from './actions'
import { actionAllowedInInput, canonicalizeCombo } from './combo'

// #46525 / #49422: send and newline are the most personal keys in a chat
// composer — an Enter-to-send default trips CJK IME users mid-composition
// (Enter is the candidate-confirm key) and every major chat client makes the
// pair configurable. These hold the registration side of that contract; the
// keydown behavior contract lives in the composer suite.
describe('composer.send / composer.newline are rebindable (#46525, #49422)', () => {
  it('registers both as rebindable actions in the composer category', () => {
    expect(keybindAction('composer.send')).toMatchObject({ category: 'composer', defaults: ['enter'] })
    expect(keybindAction('composer.newline')).toMatchObject({ category: 'composer', defaults: ['shift+enter'] })

    // No stale readonly twin rows: the panel would list the same action twice,
    // once rebindable and once fixed — exactly the dead-badge bug users hit.
    expect(KEYBIND_ACTIONS.filter(action => action.id === 'composer.send')).toHaveLength(1)
    expect(KEYBIND_ACTIONS.filter(action => action.id === 'composer.newline')).toHaveLength(1)
    expect(KEYBIND_READONLY.filter(row => row.id === 'composer.send' || row.id === 'composer.newline')).toEqual([])
  })

  it('ships the legacy default pair: Enter sends, Shift+Enter inserts a newline', () => {
    expect(defaultBindings()['composer.send']).toEqual(['enter'])
    expect(defaultBindings()['composer.newline']).toEqual(['shift+enter'])
  })

  it('keeps every shipped default chord unique across actions and readonly rows', () => {
    const shipped = new Map<string, string[]>()

    for (const action of KEYBIND_ACTIONS) {
      for (const combo of action.defaults) {
        const key = canonicalizeCombo(combo)
        shipped.set(key, [...(shipped.get(key) ?? []), action.id])
      }
    }

    // Layering declared as `passthrough` (tab slot → profile switch, soft
    // Enter → composer.send) is intentional sharing, not a conflict.
    for (const owners of [...shipped.values()].filter(ids => ids.length > 1)) {
      expect(keybindAction(owners[0]!)?.passthrough, `shared chord ${owners.join(' + ')}`).toBe(true)
    }
  })

  it('declines the composer chords in the global dispatcher while an editable target has focus', () => {
    // The rich editor resolves both from $bindings in its own keydown. A
    // modified rebind (mod+enter) that ALSO fired here would submit twice —
    // once claimed by the editor, once dispatched globally.
    expect(actionAllowedInInput('composer.send', 'mod+enter')).toBe(false)
    expect(actionAllowedInInput('composer.newline', 'enter')).toBe(false)
  })

  it('layers the shared bare-Enter default: soft focus claims it first and passes through to send', () => {
    // Dispatch order is KEYBIND_ACTIONS order. `composer.focus` must sit ahead
    // of `composer.send` and carry `passthrough`, so Enter outside the composer
    // focuses it (send declines by target gate), and a bare-Enter send binding
    // never reports the shared default chord as a conflict.
    const ids = KEYBIND_ACTIONS.map(action => action.id)

    expect(ids.indexOf('composer.focus')).toBeLessThan(ids.indexOf('composer.send'))
    expect(keybindAction('composer.focus')?.passthrough).toBe(true)
  })
})
