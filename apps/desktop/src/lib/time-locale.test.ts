import { expect, it } from 'vitest'

import { DAY, HOUR, MINUTE, relativeTime, SECOND } from './time'

it.each(['en', 'ja', 'zh', 'zh-hant', 'ar', 'ru', 'fr', 'de', 'es'])(
  'formats relative times in the selected %s locale',
  locale => {
    const format = new Intl.RelativeTimeFormat(locale, { numeric: 'auto', style: 'short' })

    for (const [unit, size] of [
      ['second', SECOND],
      ['minute', MINUTE],
      ['hour', HOUR],
      ['day', DAY]
    ] as const) {
      for (const count of [-2, 2]) {
        expect(relativeTime(count * size, 0, locale)).toBe(format.format(count, unit))
      }
    }
  }
)
