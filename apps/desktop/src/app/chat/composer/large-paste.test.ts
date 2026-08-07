import { describe, expect, it } from 'vitest'

import { LARGE_PASTE_ATTACHMENT_THRESHOLD, shouldConvertPasteToAttachment } from './large-paste'

describe('shouldConvertPasteToAttachment', () => {
  it('keeps short pastes inline', () => {
    expect(shouldConvertPasteToAttachment('hello world')).toBe(false)
    expect(shouldConvertPasteToAttachment('')).toBe(false)
  })

  it('keeps a paste exactly at the threshold inline', () => {
    expect(shouldConvertPasteToAttachment('a'.repeat(LARGE_PASTE_ATTACHMENT_THRESHOLD))).toBe(false)
  })

  it('converts a paste one character past the threshold', () => {
    expect(shouldConvertPasteToAttachment('a'.repeat(LARGE_PASTE_ATTACHMENT_THRESHOLD + 1))).toBe(true)
  })

  it('honors a custom threshold', () => {
    expect(shouldConvertPasteToAttachment('abcdef', 5)).toBe(true)
    expect(shouldConvertPasteToAttachment('abcde', 5)).toBe(false)
  })

  it('never converts when the threshold is disabled (non-positive)', () => {
    expect(shouldConvertPasteToAttachment('a'.repeat(50_000), 0)).toBe(false)
    expect(shouldConvertPasteToAttachment('a'.repeat(50_000), -1)).toBe(false)
  })

  it('rejects non-string input defensively', () => {
    expect(shouldConvertPasteToAttachment(undefined as unknown as string)).toBe(false)
    expect(shouldConvertPasteToAttachment(null as unknown as string)).toBe(false)
  })
})
