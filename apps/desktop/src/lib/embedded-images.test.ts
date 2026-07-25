import { describe, expect, it } from 'vitest'

import { extractEmbeddedImages, extractImageRefs, textWithoutImageRefs } from './embedded-images'

const SAMPLE_PNG_DATA_URL = 'data:image/png;base64,' + 'A'.repeat(120)

describe('extractEmbeddedImages', () => {
  it('returns text untouched when no data URL is present', () => {
    expect(extractEmbeddedImages('describe this')).toEqual({ cleanedText: 'describe this', images: [] })
  })

  it('lifts a bare data:image URL out of prose', () => {
    const result = extractEmbeddedImages(`describe this ${SAMPLE_PNG_DATA_URL}`)

    expect(result.cleanedText).toBe('describe this')
    expect(result.images).toEqual([SAMPLE_PNG_DATA_URL])
  })

  it('lifts a JSON-wrapped image_url envelope out of prose', () => {
    const result = extractEmbeddedImages(
      `describe this{"type":"image_url","image_url":{"url":"${SAMPLE_PNG_DATA_URL}"}}`
    )

    expect(result.cleanedText).toBe('describe this')
    expect(result.images).toEqual([SAMPLE_PNG_DATA_URL])
  })

  it('extracts multiple embedded images', () => {
    const second = 'data:image/jpeg;base64,' + 'B'.repeat(96)
    const result = extractEmbeddedImages(`first ${SAMPLE_PNG_DATA_URL} mid ${second} tail`)

    expect(result.cleanedText).toBe('first  mid  tail')
    expect(result.images).toEqual([SAMPLE_PNG_DATA_URL, second])
  })

  it('handles multi-megabyte data URLs without overflowing the JS stack', () => {
    const hugeDataUrl = 'data:image/png;base64,' + 'A'.repeat(8_000_000)
    const result = extractEmbeddedImages(`describe this ${hugeDataUrl} thanks`)

    expect(result.cleanedText).toBe('describe this  thanks')
    expect(result.images).toHaveLength(1)
    expect(result.images[0]).toHaveLength(hugeDataUrl.length)
  })
})

describe('textWithoutImageRefs', () => {
  it('leaves plain text untouched', () => {
    expect(textWithoutImageRefs('just a question')).toBe('just a question')
  })

  it('strips a single leading @image directive line', () => {
    expect(textWithoutImageRefs('@image:/tmp/cat.png\nwhat is this?')).toBe('what is this?')
  })

  it('strips multiple @image directive lines and trims', () => {
    const input = '@image:/tmp/a.png\n@image:/tmp/b.png\n  describe both  '

    expect(textWithoutImageRefs(input)).toBe('describe both')
  })

  it('does not treat an inline @image mention as a directive line', () => {
    // Only full-line leading directives are stripped, matching the gateway's
    // persist-time rewrite. A bare mention mid-prose is preserved.
    expect(textWithoutImageRefs('see @image:/tmp/cat.png here')).toBe('see @image:/tmp/cat.png here')
  })
})

describe('extractImageRefs', () => {
  it('returns the text untouched and no refs when there are no directives', () => {
    expect(extractImageRefs('a normal prompt')).toEqual({ cleanedText: 'a normal prompt', refs: [] })
  })

  it('lifts leading @image directive lines into refs and clears the text', () => {
    const result = extractImageRefs('@image:/tmp/cat.png\nwhat do you see?')

    expect(result).toEqual({ cleanedText: 'what do you see?', refs: ['@image:/tmp/cat.png'] })
  })

  it('collects multiple refs in order', () => {
    const result = extractImageRefs('@image:/tmp/a.png\n@image:/tmp/b.png\ncompare them')

    expect(result.cleanedText).toBe('compare them')
    expect(result.refs).toEqual(['@image:/tmp/a.png', '@image:/tmp/b.png'])
  })

  it('keeps only the directive lines when there is no trailing text', () => {
    const result = extractImageRefs('@image:/tmp/only.png')

    expect(result).toEqual({ cleanedText: '', refs: ['@image:/tmp/only.png'] })
  })
})
