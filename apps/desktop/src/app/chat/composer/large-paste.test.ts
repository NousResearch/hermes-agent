import { describe, expect, it } from 'vitest'

import {
  LARGE_PASTE_ATTACHMENT_THRESHOLD,
  normalizeLargePasteAttachmentThreshold,
  pasteSizeLabel,
  shouldConvertPasteToAttachment
} from './large-paste'

describe('large paste policy', () => {
  it('converts only pastes strictly past the threshold', () => {
    expect(shouldConvertPasteToAttachment('a'.repeat(LARGE_PASTE_ATTACHMENT_THRESHOLD - 1))).toBe(false)
    expect(shouldConvertPasteToAttachment('a'.repeat(LARGE_PASTE_ATTACHMENT_THRESHOLD))).toBe(false)
    expect(shouldConvertPasteToAttachment('a'.repeat(LARGE_PASTE_ATTACHMENT_THRESHOLD + 1))).toBe(true)
    expect(shouldConvertPasteToAttachment('a'.repeat(50_000), 0)).toBe(false)
  })

  it('honors custom thresholds and allows conversion to be disabled', () => {
    expect(shouldConvertPasteToAttachment('a'.repeat(11_700), 50_000)).toBe(false)
    expect(shouldConvertPasteToAttachment('a'.repeat(50_000), 50_000)).toBe(false)
    expect(shouldConvertPasteToAttachment('a'.repeat(50_001), 50_000)).toBe(true)
    expect(shouldConvertPasteToAttachment('a'.repeat(100_001), 0)).toBe(false)
    expect(shouldConvertPasteToAttachment('a'.repeat(100_000), 100_000)).toBe(false)
    expect(shouldConvertPasteToAttachment('a'.repeat(100_001), 100_000)).toBe(true)
  })

  it.each([0, '0', 1, '50000', 100_000])('accepts a threshold of %s', value => {
    expect(normalizeLargePasteAttachmentThreshold(value)).toBe(Number(value))
  })

  it.each([undefined, null, '', ' ', 'invalid', false, [], -1, 1.5, NaN, Infinity, 100_001])(
    'retains the default for invalid preference %s',
    value => {
      expect(normalizeLargePasteAttachmentThreshold(value)).toBe(LARGE_PASTE_ATTACHMENT_THRESHOLD)
    }
  )

  it('labels the chip by encoded byte size, not character count', () => {
    expect(pasteSizeLabel('a'.repeat(512))).toBe('512 B')
    expect(pasteSizeLabel('\u00e9'.repeat(1024))).toBe('2.0 KB')
  })
})
