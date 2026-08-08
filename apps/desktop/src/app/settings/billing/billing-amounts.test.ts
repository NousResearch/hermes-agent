import { afterEach, describe, expect, it, vi } from 'vitest'

import { formatMoney } from './billing-amounts'

afterEach(() => {
  vi.restoreAllMocks()
})

describe('formatMoney', () => {
  it('pins hard-coded USD copy to en-US formatting', () => {
    const OriginalNumberFormat = Intl.NumberFormat
    const seenLocales: Array<Intl.LocalesArgument | undefined> = []

    vi.spyOn(Intl, 'NumberFormat').mockImplementation(
      class {
        constructor(locales?: Intl.LocalesArgument, options?: Intl.NumberFormatOptions) {
          seenLocales.push(locales)

          return new OriginalNumberFormat(locales ?? 'en-DE', options)
        }
      } as typeof Intl.NumberFormat
    )

    expect(formatMoney(25)).toBe('$25')
    expect(seenLocales).toEqual(['en-US'])
  })
})
