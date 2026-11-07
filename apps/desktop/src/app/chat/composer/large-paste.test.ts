import { describe, expect, it } from 'vitest'

import {
  LARGE_PASTE_ATTACHMENT_THRESHOLD,
  LARGE_PASTE_TITLE_PREVIEW_CHARS,
  pasteSizeLabel,
  pasteTitlePreview,
  shouldConvertPasteToAttachment
} from './large-paste'

/** A UTF-16 string whose final code unit is an unpaired high surrogate. */
function endsWithLoneHighSurrogate(value: string): boolean {
  const last = value.charCodeAt(value.length - 1)

  return last >= 0xd8_00 && last <= 0xdb_ff
}

describe('large paste policy', () => {
  it('converts only pastes strictly past the threshold', () => {
    expect(shouldConvertPasteToAttachment('a'.repeat(LARGE_PASTE_ATTACHMENT_THRESHOLD))).toBe(false)
    expect(shouldConvertPasteToAttachment('a'.repeat(LARGE_PASTE_ATTACHMENT_THRESHOLD + 1))).toBe(true)
    expect(shouldConvertPasteToAttachment('a'.repeat(50_000), 0)).toBe(false)
  })

  it('labels the chip by encoded byte size, not character count', () => {
    expect(pasteSizeLabel('a'.repeat(512))).toBe('512 B')
    expect(pasteSizeLabel('é'.repeat(1024))).toBe('2.0 KB')
  })
})

describe('paste title preview', () => {
  it('returns short text unchanged, including text of exactly the unit budget', () => {
    expect(pasteTitlePreview('short paste')).toBe('short paste')
    expect(pasteTitlePreview('a'.repeat(LARGE_PASTE_TITLE_PREVIEW_CHARS))).toBe(
      'a'.repeat(LARGE_PASTE_TITLE_PREVIEW_CHARS)
    )
  })

  it('drops an astral character whose surrogate pair straddles the cut (#150281)', () => {
    // The reported shape: the boundary landed between the two code units of 🧩, and the
    // half pair that survived became `title_preview` — then aborted the admission's bind.
    const preview = pasteTitlePreview(`${'a'.repeat(LARGE_PASTE_TITLE_PREVIEW_CHARS - 1)}🧩 tail`)

    expect(preview).toBe('a'.repeat(LARGE_PASTE_TITLE_PREVIEW_CHARS - 1))
    expect(endsWithLoneHighSurrogate(preview)).toBe(false)
    expect(preview.length).toBeLessThanOrEqual(LARGE_PASTE_TITLE_PREVIEW_CHARS)
  })

  it('never splits a ZWJ sequence at the cut', () => {
    const family = '👨‍👩‍👧‍👦'
    const preview = pasteTitlePreview(`${'a'.repeat(LARGE_PASTE_TITLE_PREVIEW_CHARS - 2)}${family}`)

    expect(preview).toBe('a'.repeat(LARGE_PASTE_TITLE_PREVIEW_CHARS - 2))
    expect(preview).not.toContain('\u200d')
  })

  it('drops a cluster that straddles the cut instead of keeping its dangling tail', () => {
    // The cluster is 11 units long, so the cut can land inside it while the retained units still
    // end on a complete pair — no lone surrogate for a "drop the half pair" rule to detect, but a
    // ZWJ fragment (`👨‍👩` out of `👨‍👩‍👧‍👦`) in the stored title preview.
    const family = '👨‍👩‍👧‍👦'
    const kept = `${'a'.repeat(995)}${family}`

    expect(kept.slice(0, LARGE_PASTE_TITLE_PREVIEW_CHARS)).toBe(`${'a'.repeat(995)}👨‍👩`)
    expect(pasteTitlePreview(kept)).toBe('a'.repeat(995))
  })

  it('keeps a trailing astral character that still fits in the budget', () => {
    const preview = pasteTitlePreview(`${'a'.repeat(LARGE_PASTE_TITLE_PREVIEW_CHARS - 2)}🧩`)

    expect(preview.endsWith('🧩')).toBe(true)
    expect(preview.length).toBe(LARGE_PASTE_TITLE_PREVIEW_CHARS)
  })

  it('caps a multi-megabyte paste without ending on a half pair', () => {
    const preview = pasteTitlePreview(`${'log line\n'.repeat(200_000)}🧩`)

    expect(preview.length).toBeGreaterThan(0)
    expect(preview.length).toBeLessThanOrEqual(LARGE_PASTE_TITLE_PREVIEW_CHARS)
    expect(endsWithLoneHighSurrogate(preview)).toBe(false)
  })

  it('honours an explicit budget and refuses a non-positive one', () => {
    expect(pasteTitlePreview('abcdef', 4)).toBe('abcd')
    expect(pasteTitlePreview('a🧩b', 2)).toBe('a')
    expect(pasteTitlePreview('abcdef', 0)).toBe('')
  })
})
