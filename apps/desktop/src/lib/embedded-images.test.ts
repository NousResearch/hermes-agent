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

  it('lifts a JSON-wrapped envelope containing whitespace out of prose', () => {
    const result = extractEmbeddedImages(
      `before {"type": "image_url", "image_url": {"url": "${SAMPLE_PNG_DATA_URL}"}} after`
    )

    expect(result.cleanedText).toBe('before  after')
    expect(result.images).toEqual([SAMPLE_PNG_DATA_URL])
  })

  it('ignores base64 runs shorter than 64 characters', () => {
    const short = 'data:image/png;base64,' + 'A'.repeat(63)
    const result = extractEmbeddedImages(`inline ${short} stays`)

    expect(result.cleanedText).toBe(`inline ${short} stays`)
    expect(result.images).toEqual([])
  })

  it('extracts a multi-megabyte payload without overflowing the regexp stack', () => {
    const huge = 'data:image/png;base64,' + 'A'.repeat(6_000_000)
    const result = extractEmbeddedImages(`screenshot incoming ${huge} done`)

    expect(result.cleanedText).toBe('screenshot incoming  done')
    expect(result.images).toEqual([huge])
  })
})
