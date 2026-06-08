import { describe, expect, it } from 'vitest'

import { shouldBlockPlainTextPaste } from './clipboard-paste'

describe('shouldBlockPlainTextPaste', () => {
  it('blocks PNG binary bytes exposed as text by the WSL clipboard bridge', () => {
    const pngBytesAsText = '\x89PNG\r\n\x1A\n\x00\x00\x00\rIHDR\x00\x00\x09\xC8'

    expect(shouldBlockPlainTextPaste(pngBytesAsText)).toBe(true)
  })

  it('blocks PNG bytes decoded with a replacement character by Chromium', () => {
    const pngText = '\uFFFDPNG\r\n\x1A\n\x00\x00\x00\rIHDR\x00\x00\x09\xC8'

    expect(shouldBlockPlainTextPaste(pngText)).toBe(true)
  })

  it('does not block normal text paste', () => {
    expect(shouldBlockPlainTextPaste('please check this invoice')).toBe(false)
  })

  it('does not block normal technical text about PNG chunks', () => {
    expect(shouldBlockPlainTextPaste('A PNG file starts with a signature and has an IHDR chunk.')).toBe(false)
  })

  it('does not block a small number of replacement characters in text', () => {
    expect(shouldBlockPlainTextPaste('encoding note: � appears when a glyph cannot be decoded')).toBe(false)
  })

  it('blocks mostly-control-character binary text', () => {
    const binaryText = '\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0E\x0F\x10\x11\x12\x13\x14hello'

    expect(shouldBlockPlainTextPaste(binaryText)).toBe(true)
  })
})
