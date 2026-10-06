import { describe, expect, it } from 'vitest'

import { resolveSessionRowClick } from './session-row-gesture'

const NO_MODS = { altKey: false, ctrlKey: false, metaKey: false, shiftKey: false }
const WINDOW_OK = { canOpenWindow: true }
const NO_WINDOW = { canOpenWindow: false }

describe('resolveSessionRowClick', () => {
  it('resumes on a plain click', () => {
    expect(resolveSessionRowClick(NO_MODS, WINDOW_OK)).toBe('resume')
  })

  it('pins on ⇧-click', () => {
    expect(resolveSessionRowClick({ ...NO_MODS, shiftKey: true }, WINDOW_OK)).toBe('pin')
  })

  it('opens a new tab on ⌘/⌃-click', () => {
    expect(resolveSessionRowClick({ ...NO_MODS, metaKey: true }, WINDOW_OK)).toBe('newTab')
    expect(resolveSessionRowClick({ ...NO_MODS, ctrlKey: true }, WINDOW_OK)).toBe('newTab')
    expect(resolveSessionRowClick({ ...NO_MODS, ctrlKey: true }, NO_WINDOW)).toBe('newTab')
  })

  it('opens a new window on ⌘/⌃+⇧-click when supported', () => {
    expect(resolveSessionRowClick({ ...NO_MODS, metaKey: true, shiftKey: true }, WINDOW_OK)).toBe('newWindow')
    expect(resolveSessionRowClick({ ...NO_MODS, ctrlKey: true, shiftKey: true }, WINDOW_OK)).toBe('newWindow')
  })

  it('falls back to a new tab for ⌘/⌃+⇧-click when windows are unavailable (web embed)', () => {
    expect(resolveSessionRowClick({ ...NO_MODS, metaKey: true, shiftKey: true }, NO_WINDOW)).toBe('newTab')
  })

  // The regression this whole module guards: ⌥+⇧ sets shiftKey too, so a naive
  // "check shiftKey first" would swallow archive into "pin".
  it('archives on ⌥+⇧-click', () => {
    expect(resolveSessionRowClick({ ...NO_MODS, altKey: true, shiftKey: true }, WINDOW_OK)).toBe('archive')
  })

  it('archives regardless of window support (archive needs no standalone window)', () => {
    expect(resolveSessionRowClick({ ...NO_MODS, altKey: true, shiftKey: true }, NO_WINDOW)).toBe('archive')
  })

  it('does not archive on ⌥-click alone', () => {
    expect(resolveSessionRowClick({ ...NO_MODS, altKey: true }, WINDOW_OK)).toBe('resume')
  })
})
