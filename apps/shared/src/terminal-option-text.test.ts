import { describe, expect, it } from 'vitest'

import { macOptionComposedText } from './terminal-option-text'

const keydown = (key: string, init: Partial<{ ctrlKey: boolean; metaKey: boolean; type: string }> = {}) => ({
  altKey: true,
  ctrlKey: false,
  key,
  metaKey: false,
  type: 'keydown',
  ...init
})

// Everything a US (ABC) layout produces with Option and Option+Shift.
const US_OPTION_GLYPHS = [...'¡™£¢∞§¶•ªº–≠œ∑®†¥øπ“‘«åß∂ƒ©˙∆˚¬…æΩ≈ç√∫µ≤≥÷⁄€‹›ﬁﬂ‡°·‚—±Œ„‰ˇÁØ∏”’»ÅÍÎÏ˝ÓÔÒÚÆ¸˛Ç◊ı˜Â¯˘¿', 'Dead']

describe('macOptionComposedText', () => {
  it('passes only composed ASCII punctuation, on macOS keydowns with Option alone', () => {
    for (const glyph of US_OPTION_GLYPHS) {
      expect(macOptionComposedText(keydown(glyph), true)).toBeNull()
    }

    for (const char of '@|[]{}\\~') {
      expect(macOptionComposedText(keydown(char), true)).toBe(char)
      expect(macOptionComposedText(keydown(char), false)).toBeNull()
      expect(macOptionComposedText(keydown(char, { metaKey: true }), true)).toBeNull()
      expect(macOptionComposedText(keydown(char, { ctrlKey: true }), true)).toBeNull()
      expect(macOptionComposedText(keydown(char, { type: 'keypress' }), true)).toBeNull()
    }

    for (const char of 'aZ09 ') {
      expect(macOptionComposedText(keydown(char), true)).toBeNull()
    }
  })
})
