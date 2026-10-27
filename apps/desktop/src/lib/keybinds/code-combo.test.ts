import { describe, expect, it } from 'vitest'

import { defaultBindings } from './actions'
import { canonicalizeCombo, codeComboFromEvent, comboFromEvent, dispatchCombo } from './combo'

// Host-native: jsdom never reports a Mac platform, so `IS_MAC` is false here —
// Ctrl folds to `mod` and shipped `ctrl+…` defaults canonicalize to `mod+…`.
// Mac-only branches are not faked.

function keydown(init: KeyboardEventInit): KeyboardEvent {
  return new KeyboardEvent('keydown', init)
}

// The live dispatch index, built the way `$comboIndex` builds it.
function defaultIndex(): Map<string, string[]> {
  const index = new Map<string, string[]>()

  for (const [actionId, combos] of Object.entries(defaultBindings())) {
    for (const combo of combos) {
      const key = canonicalizeCombo(combo)
      const ids = index.get(key)

      if (ids) {
        ids.push(actionId)
      } else {
        index.set(key, [actionId])
      }
    }
  }

  return index
}

// Both resolutions of one synthetic keydown; the char combo is the dispatcher's
// precondition, so a keydown that does not resolve fails loudly here.
function resolutionsOf(event: KeyboardEvent): { combo: string; codeCombo: string | null } {
  const combo = comboFromEvent(event)

  if (combo == null) {
    throw new Error(`keydown did not resolve to a combo: key=${event.key} code=${event.code}`)
  }

  return { codeCombo: codeComboFromEvent(event), combo }
}

// #137849: the terminal toggle (Ctrl+`, `mod+`` off macOS) must resolve on
// layouts where the physical Backquote key types a different character. The
// dispatcher resolves the character combo first and falls back to the
// physical-code combo only when the character one is unbound — so Czech/Slovak
// (';') and pt-BR ("'") keydowns reach the toggle, while Italian ('\') keeps
// flip-panes because its character combo IS bound.

describe('dispatchCombo — character priority, physical-code fallback', () => {
  const index = defaultIndex()

  it('rescues the terminal toggle when Backquote types an unbound shipped token', () => {
    // Czech/Slovak ';' and pt-BR "'": the character combos (mod+; / mod+') are
    // unbound, so the physical-code twin (mod+`) wins.
    for (const key of [';', "'"]) {
      const { combo, codeCombo } = resolutionsOf(keydown({ code: 'Backquote', ctrlKey: true, key }))
      const dispatch = dispatchCombo(index, combo, codeCombo)

      expect(dispatch?.combo).toBe('mod+`')
      expect(dispatch?.actionIds).toContain('view.showTerminal')
    }
  })

  it("keeps flip-panes for Italian Backquote ('\\'): a bound character combo is never re-stolen", () => {
    const { combo, codeCombo } = resolutionsOf(keydown({ code: 'Backquote', ctrlKey: true, key: '\\' }))
    const dispatch = dispatchCombo(index, combo, codeCombo)

    expect(dispatch?.combo).toBe('mod+\\')
    expect(dispatch?.actionIds).toContain('view.flipPanes')
    expect(dispatch?.actionIds).not.toContain('view.showTerminal')
  })

  it('keeps character priority for Dvorak punctuation chords that are bound (#73015)', () => {
    // Dvorak puts "." on the physical V key — ⌘. must stay the command center.
    const dot = resolutionsOf(keydown({ code: 'KeyV', key: '.', metaKey: true }))
    const dotDispatch = dispatchCombo(index, dot.combo, dot.codeCombo)

    expect(dotDispatch?.combo).toBe('mod+.')
    expect(dotDispatch?.actionIds).toContain('nav.commandCenter')

    // Dvorak "/" lives on the physical [ key — ⌘/ opens the keybinds panel.
    const slash = resolutionsOf(keydown({ code: 'BracketLeft', key: '/', metaKey: true }))
    const slashDispatch = dispatchCombo(index, slash.combo, slash.codeCombo)

    expect(slashDispatch?.combo).toBe('mod+/')
    expect(slashDispatch?.actionIds).toContain('keybinds.openPanel')
  })

  it('falls through when neither resolution is bound (AltGr-only backticks)', () => {
    // A literal ` reachable only via AltGr adds 'alt' to BOTH resolutions, so
    // neither can match the plain 'mod+`' binding — the AltGr route stays open
    // as a follow-up (see the PR's decision note), it does not regress.
    const { combo, codeCombo } = resolutionsOf(keydown({ code: 'Quote', ctrlKey: true, altKey: true, key: '`' }))

    expect(dispatchCombo(index, combo, codeCombo)).toBeNull()
  })
})

describe('codeComboFromEvent — the physical-code twin of comboFromEvent', () => {
  it('resolves through event.code whatever the layout types on that key', () => {
    expect(codeComboFromEvent(keydown({ code: 'Backquote', ctrlKey: true, key: ';' }))).toBe('mod+`')
    expect(codeComboFromEvent(keydown({ code: 'Backquote', ctrlKey: true, key: "'" }))).toBe('mod+`')
    expect(codeComboFromEvent(keydown({ code: 'Backquote', ctrlKey: true, key: '\\' }))).toBe('mod+`')
    // German: the Backquote key is a dead key.
    expect(codeComboFromEvent(keydown({ code: 'Backquote', ctrlKey: true, key: 'Dead' }))).toBe('mod+`')
    // Dvorak: the physical V key types '.'.
    expect(codeComboFromEvent(keydown({ code: 'KeyV', key: '.', metaKey: true }))).toBe('mod+v')
  })

  it('shares the guard rails: modifiers, composition and IME keys never resolve', () => {
    expect(codeComboFromEvent(keydown({ code: 'ControlLeft', key: 'Control' }))).toBeNull()
    expect(codeComboFromEvent(keydown({ code: 'KeyN', isComposing: true, key: 'n' }))).toBeNull()
    expect(codeComboFromEvent(keydown({ code: 'KeyW', key: 'Process' }))).toBeNull()
    // Legacy IME-synthesized modifier names on regular key codes.
    expect(codeComboFromEvent(keydown({ code: 'KeyW', key: 'Control' }))).toBeNull()
  })
})
