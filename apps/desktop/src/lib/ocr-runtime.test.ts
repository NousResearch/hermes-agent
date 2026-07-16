import { describe, expect, it } from 'vitest'

import { defaultOcrLanguage } from './ocr-runtime'

describe('OCR locale selection', () => {
  it.each([
    ['en-US', 'eng'],
    ['ja-JP', 'jpn'],
    ['zh-CN', 'chi_sim'],
    ['zh-Hant-TW', 'chi_tra']
  ] as const)('maps %s to its bundled language pack', (locale, expected) => {
    expect(defaultOcrLanguage(locale)).toBe(expected)
  })
})
