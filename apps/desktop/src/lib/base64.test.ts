import { describe, expect, it } from 'vitest'

import { bytesToBase64 } from './base64'

describe('bytesToBase64', () => {
  it('encodes ArrayBuffer and Uint8Array inputs through the same path', () => {
    const bytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47])

    expect(bytesToBase64(bytes)).toBe('iVBORw==')
    expect(bytesToBase64(bytes.buffer)).toBe('iVBORw==')
  })
})
