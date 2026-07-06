import { describe, expect, it } from 'vitest'

import { resolveSessionRowClick } from './session-row-gesture'

const NO_MODS = { ctrlKey: false, metaKey: false, shiftKey: false }
const WINDOW_OK = { canOpenWindow: true }
const NO_WINDOW = { canOpenWindow: false }

describe('resolveSessionRowClick', () => {
  it('resumes on a plain click', () => {
    expect(resolveSessionRowClick(NO_MODS, WINDOW_OK)).toBe('resume')
  })

  it('pins on ⇧-click', () => {
    expect(resolveSessionRowClick({ ...NO_MODS, shiftKey: true }, WINDOW_OK)).toBe('pin')
  })

  it('opens a new window on ⌘/⌃-click when supported', () => {
    expect(resolveSessionRowClick({ ...NO_MODS, metaKey: true }, WINDOW_OK)).toBe('newWindow')
    expect(resolveSessionRowClick({ ...NO_MODS, ctrlKey: true }, WINDOW_OK)).toBe('newWindow')
  })

  it('falls back to resume for ⌘/⌃-click when windows are unavailable (web embed)', () => {
    expect(resolveSessionRowClick({ ...NO_MODS, metaKey: true }, NO_WINDOW)).toBe('resume')
  })

  // The regression this whole module guards: ⌘/⌃+⇧ sets shiftKey AND a primary
  // modifier, so a naive "check shiftKey first" would archive-as-pin.
  it('archives on ⌘+⇧-click (mac) and ⌃+⇧-click (win/linux)', () => {
    expect(resolveSessionRowClick({ ctrlKey: false, metaKey: true, shiftKey: true }, WINDOW_OK)).toBe('archive')
    expect(resolveSessionRowClick({ ctrlKey: true, metaKey: false, shiftKey: true }, WINDOW_OK)).toBe('archive')
  })

  it('archives regardless of window support (archive needs no standalone window)', () => {
    expect(resolveSessionRowClick({ ctrlKey: true, metaKey: false, shiftKey: true }, NO_WINDOW)).toBe('archive')
  })
})
