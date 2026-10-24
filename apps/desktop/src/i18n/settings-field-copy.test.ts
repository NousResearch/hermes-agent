import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from './catalog'
import type { BundledLocale } from './types'

describe('settings field label/description shape', () => {
  // config-field.tsx renders fieldLabels as ListRow's compact title and fieldDescriptions as the
  // helper text below. A locale that pastes the long helper into the label slot turns the settings
  // row's title into a multi-sentence paragraph while leaving the real description stale — the
  // exact regression from #134219 (ar/ja/ru/zh/zh-hant all did it for modelContextLength). A label
  // rendered as a compact title can never legitimately be longer than its own description.
  it('keeps every modelContextLength label shorter than its description in every locale', () => {
    const locales = Object.keys(TRANSLATIONS) as BundledLocale[]

    for (const locale of locales) {
      const { fieldLabels, fieldDescriptions } = TRANSLATIONS[locale].settings
      const label = fieldLabels.modelContextLength
      const description = fieldDescriptions.modelContextLength

      expect(label.length, `${locale} modelContextLength label vs description length`).toBeLessThan(
        description.length
      )
    }
  })
})
