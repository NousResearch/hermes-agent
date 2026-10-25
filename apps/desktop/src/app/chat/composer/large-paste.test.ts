import { describe, expect, it } from 'vitest'

import {
  INLINE_PASTE_SHORTCUT_WINDOW_MS,
  isInlinePasteRequested,
  isInlinePasteShortcut,
  LARGE_PASTE_ATTACHMENT_THRESHOLD,
  pasteSizeLabel,
  shouldConvertPasteToAttachment
} from './large-paste'

describe('large paste policy', () => {
  it('converts only pastes strictly past the threshold', () => {
    expect(shouldConvertPasteToAttachment('a'.repeat(LARGE_PASTE_ATTACHMENT_THRESHOLD))).toBe(false)
    expect(shouldConvertPasteToAttachment('a'.repeat(LARGE_PASTE_ATTACHMENT_THRESHOLD + 1))).toBe(true)
    expect(shouldConvertPasteToAttachment('a'.repeat(50_000), 0)).toBe(false)
  })

  it('labels the chip by encoded byte size, not character count', () => {
    expect(pasteSizeLabel('a'.repeat(512))).toBe('512 B')
    expect(pasteSizeLabel('\u00e9'.repeat(1024))).toBe('2.0 KB')
  })
})

describe('inline paste shortcut', () => {
  const key = (init: KeyboardEventInit) => new KeyboardEvent('keydown', init)

  it('recognises ⌘⇧V and Ctrl+Shift+V, and nothing else', () => {
    expect(isInlinePasteShortcut(key({ key: 'v', metaKey: true, shiftKey: true }))).toBe(true)
    expect(isInlinePasteShortcut(key({ key: 'V', ctrlKey: true, shiftKey: true }))).toBe(true)
    expect(isInlinePasteShortcut(key({ key: 'v', ctrlKey: true }))).toBe(false)
    expect(isInlinePasteShortcut(key({ key: 'v', shiftKey: true }))).toBe(false)
    expect(isInlinePasteShortcut(key({ key: 'v', altKey: true, ctrlKey: true, shiftKey: true }))).toBe(false)
    expect(isInlinePasteShortcut(key({ key: 'k', ctrlKey: true, shiftKey: true }))).toBe(false)
  })

  it('arms only for a short window, so a chord with nothing to paste never leaks into the next ⌘V', () => {
    expect(isInlinePasteRequested(0, 1_000)).toBe(false)
    expect(isInlinePasteRequested(1_000, 1_000 + INLINE_PASTE_SHORTCUT_WINDOW_MS - 1)).toBe(true)
    expect(isInlinePasteRequested(1_000, 1_000 + INLINE_PASTE_SHORTCUT_WINDOW_MS)).toBe(false)
  })
})
