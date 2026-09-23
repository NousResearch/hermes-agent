import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from './catalog'
import type { Locale } from './types'

const SIDEBAR_FILTER_MENU_KEYS = [
  'grouping',
  'ordering',
  'show',
  'filters',
  'status',
  'pullRequest',
  'profile',
  'project',
  'resetToDefaults',
  'expandAll',
  'collapseAll',
  'updated',
  'created',
  'tokens',
  'cost',
  'manual',
  'preview',
  'pr',
  'open',
  'draft',
  'merged',
  'closed',
  'noPr',
  'needsInput',
  'workingStatus',
  'unread',
  'draftStatus',
  'idle',
  'archived',
  'inboxStyle'
] as const

const LOCALES = ['en', 'zh', 'zh-hant', 'ja', 'ar', 'ru'] satisfies Locale[]

const filterMenuOf = (locale: Locale): Record<string, string> =>
  TRANSLATIONS[locale].sidebar.filterMenu as unknown as Record<string, string>

describe('sidebar.filterMenu locale parity', () => {
  it('declares a non-empty string for every key in every locale', () => {
    for (const locale of LOCALES) {
      const menu = filterMenuOf(locale)

      for (const key of SIDEBAR_FILTER_MENU_KEYS) {
        expect(typeof menu[key], `${locale}.sidebar.filterMenu.${key}`).toBe('string')
        expect(menu[key].trim(), `${locale}.sidebar.filterMenu.${key} is blank`).not.toBe('')
      }
    }
  })

  it('keeps every locale on the same key set as en', () => {
    const enKeys = Object.keys(filterMenuOf('en')).sort()

    for (const locale of LOCALES) {
      expect(Object.keys(filterMenuOf(locale)).sort(), locale).toEqual(enKeys)
    }
  })

  it.each(LOCALES.filter(locale => locale !== 'en'))(
    '%s translates the menu headings instead of falling back to English',
    locale => {
      const enMenu = filterMenuOf('en')
      const menu = filterMenuOf(locale as Locale)

      for (const key of ['grouping', 'ordering', 'filters']) {
        expect(menu[key], `${locale}.sidebar.filterMenu.${key} still says "${enMenu[key]}"`).not.toBe(enMenu[key])
      }
    }
  )
})
