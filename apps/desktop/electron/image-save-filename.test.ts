import { describe, expect, it } from 'vitest'

import { imageSaveFilename } from './image-save-filename'

describe('imageSaveFilename', () => {
  it('prefers a supplied original filename when the image source is a data URL', () => {
    expect(imageSaveFilename('data:image/png;base64,ZHVtbXk=', 'generated-image.png', '.png')).toBe(
      'generated-image.png'
    )
  })

  it('uses a unique timestamped fallback instead of image.png when no filename survives', () => {
    expect(
      imageSaveFilename(
        'data:image/png;base64,ZHVtbXk=',
        undefined,
        '.png',
        new Date('2026-07-15T14:35:22.123Z'),
        'a1b2c3'
      )
    ).toBe('hermes-image-20260715-143522-123-a1b2c3.png')
  })

  it('preserves an extensionless source basename and appends the MIME extension', () => {
    expect(imageSaveFilename('https://example.com/generated-image', undefined, '.png')).toBe('generated-image.png')
  })

  it.each([
    ['../../CON.png', '_CON.png'],
    ['photo...   ', 'photo.png'],
    ['encoded%2Fname%3F.png', 'name-.png'],
    ['bad\u0000name.png', 'badname.png']
  ])('sanitizes %s to a cross-platform basename', (input, expected) => {
    expect(imageSaveFilename('data:image/png;base64,ZHVtbXk=', input, '.png')).toBe(expected)
  })

  it('bounds oversized filenames while preserving their extension', () => {
    const filename = imageSaveFilename('data:image/png;base64,ZHVtbXk=', `${'a'.repeat(400)}.png`, '.png')

    expect(filename).toHaveLength(240)
    expect(filename.endsWith('.png')).toBe(true)
  })

  it.each([`${'😀'.repeat(100)}.png`, `${'界'.repeat(100)}.png`])(
    'bounds multibyte filename %s without splitting a code point',
    input => {
      const filename = imageSaveFilename('data:image/png;base64,ZHVtbXk=', input, '.png')

      expect(Buffer.byteLength(filename, 'utf8')).toBeLessThanOrEqual(240)
      expect(filename.endsWith('.png')).toBe(true)
      expect(filename).not.toContain('\uFFFD')
    }
  )

  it.each([
    ['report.html', 'report.png'],
    ['.hidden.png', 'hidden.png'],
    [`photo\u202Egnp.jpg`, 'photognp.png']
  ])('normalizes deceptive or MIME-mismatched name %s', (input, expected) => {
    expect(imageSaveFilename('data:image/png;base64,ZHVtbXk=', input, '.png')).toBe(expected)
  })

  it('keeps a URL basename through query strings and malformed escapes', () => {
    expect(imageSaveFilename('https://example.com/photo.png?token=abc', undefined, '.png')).toBe('photo.png')
    expect(imageSaveFilename('https://example.com/bad%E0%A4%A.png', undefined, '.png')).toBe('bad%E0%A4%A.png')
  })
})
