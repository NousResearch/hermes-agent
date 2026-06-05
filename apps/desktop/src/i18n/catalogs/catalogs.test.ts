import { describe, expect, it } from 'vitest'

import { en } from './en'
import { zh } from './zh'

const ZH_PARITY_EXCEPTIONS: Record<string, string> = {}

describe('desktop i18n catalogs', () => {
  it('keeps Simplified Chinese catalog coverage in parity with English', () => {
    const missingZhKeys = Object.keys(en).filter(key => !(key in zh) && !(key in ZH_PARITY_EXCEPTIONS))

    expect(missingZhKeys).toEqual([])
  })

  it('keeps the parity exception whitelist explicit and reasoned', () => {
    for (const [key, reason] of Object.entries(ZH_PARITY_EXCEPTIONS)) {
      expect(key in en).toBe(true)
      expect(reason.trim().length).toBeGreaterThan(0)
    }
  })

  it('keeps Simplified Chinese keys anchored to English source keys', () => {
    const extraZhKeys = Object.keys(zh).filter(key => !(key in en))

    expect(extraZhKeys).toEqual([])
  })
})
