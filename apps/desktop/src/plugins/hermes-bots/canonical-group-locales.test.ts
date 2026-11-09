import { expect, it } from 'vitest'

import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'

it.each([
  ['ja', 'ボット'],
  ['zh', '机器人'],
  ['zh-hant', '機器人']
] as const)('keeps translated bot terms joined to surrounding text in %s', (locale, term) => {
  for (const [key, text] of Object.entries(CANONICAL_GROUP_LOCALES[locale])) {
    expect(text, `${locale}.${key}`).not.toMatch(new RegExp(`\\s${term}|${term}\\s`))
  }
})
