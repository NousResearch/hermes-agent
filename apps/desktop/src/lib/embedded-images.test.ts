import { describe, expect, it } from 'vitest'

import { extractEmbeddedImages } from './embedded-images'

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

  it('caps extracted images and removes overflow data URLs from visible text', () => {
    const urls = Array.from({ length: 4 }, (_, index) => `data:image/png;base64,${String(index).repeat(96)}`)
    const result = extractEmbeddedImages(urls.join(' '), { maxImages: 2 })

    expect(result.images).toEqual(urls.slice(0, 2))
    expect(result.cleanedText).toBe('')
  })

  it('leaves short invalid data:image URLs untouched', () => {
    const short = 'data:image/png;base64,AAAA'

    expect(extractEmbeddedImages(`show ${short}`)).toEqual({ cleanedText: `show ${short}`, images: [] })
  })

  it('handles multi-megabyte image data URLs without recursive regex overflow', () => {
    const large = 'data:image/jpeg;base64,' + 'A'.repeat(6_000_000)
    const result = extractEmbeddedImages(`before ${large} after`)

    expect(result.cleanedText).toBe('before  after')
    expect(result.images).toHaveLength(1)
    expect(result.images[0]).toHaveLength(large.length)
  })

  it('handles multi-megabyte JSON-wrapped image_url data URLs', () => {
    const large = 'data:image/jpeg;base64,' + 'A'.repeat(6_000_000)
    const result = extractEmbeddedImages(`before {"type":"image_url","image_url":{"url":"${large}"}} after`)

    expect(result.cleanedText).toBe('before  after')
    expect(result.images).toEqual([large])
  })
})
