import { normalize } from '@/lib/text'

import type { Locale } from './types'

export const DEFAULT_LOCALE: Locale = 'en'

export const LOCALE_OPTIONS = [
  {
    id: 'en',
    name: 'English',
    englishName: 'English',
    configValue: 'en'
  },
  {
    id: 'zh',
    name: '简体中文',
    englishName: 'Simplified Chinese',
    configValue: 'zh'
  },
  {
    id: 'zh-hant',
    name: '繁體中文',
    englishName: 'Traditional Chinese',
    configValue: 'zh-hant'
  },
  {
    id: 'ja',
    name: '日本語',
    englishName: 'Japanese',
    configValue: 'ja'
  }
] as const satisfies readonly { configValue: string; englishName: string; id: Locale; name: string }[]

// `name` is the endonym (native name) shown in the picker so users recognize
// their language regardless of the current UI language. No country flags:
// languages are not countries. `englishName` is search-only (not shown) so an
// English speaker can type "japanese"/"traditional" to filter the list.
export const LOCALE_META: Record<Locale, { name: string; englishName: string }> = Object.fromEntries(
  LOCALE_OPTIONS.map(locale => [locale.id, { name: locale.name, englishName: locale.englishName }])
) as Record<Locale, { name: string; englishName: string }>

const LOCALE_ALIASES: Record<string, Locale> = {
  en: 'en',
  'en-us': 'en',
  en_us: 'en',
  zh: 'zh',
  'zh-cn': 'zh',
  zh_cn: 'zh',
  'zh-hans': 'zh',
  zh_hans: 'zh',
  'zh-hans-cn': 'zh',
  zh_hans_cn: 'zh',
  'zh-tw': 'zh-hant',
  zh_tw: 'zh-hant',
  'zh-hk': 'zh-hant',
  zh_hk: 'zh-hant',
  'zh-mo': 'zh-hant',
  zh_mo: 'zh-hant',
  'zh-hant': 'zh-hant',
  zh_hant: 'zh-hant',
  'zh-hant-tw': 'zh-hant',
  zh_hant_tw: 'zh-hant',
  'zh-hant-hk': 'zh-hant',
  zh_hant_hk: 'zh-hant',
  ja: 'ja',
  'ja-jp': 'ja',
  ja_jp: 'ja'
}

export function isLocale(value: unknown): value is Locale {
  return typeof value === 'string' && LOCALE_OPTIONS.some(locale => locale.id === value)
}

export function normalizeLocale(value: unknown): Locale {
  if (typeof value !== 'string') {
    return DEFAULT_LOCALE
  }

  return LOCALE_ALIASES[normalize(value)] ?? DEFAULT_LOCALE
}

export function isSupportedLocaleValue(value: unknown): boolean {
  return typeof value === 'string' && LOCALE_ALIASES[normalize(value)] != null
}

export function localeConfigValue(locale: Locale): string {
  return LOCALE_OPTIONS.find(item => item.id === locale)?.configValue ?? DEFAULT_LOCALE
}

// Given the platform's preferred locales in priority order (most-preferred
// first, e.g. `navigator.languages`), return the first one that maps to a
// supported UI locale, or null when none do. Used to seed the initial UI
// language from the operating-system locale on first launch, before the user
// has explicitly picked one.
export function matchPreferredLocale(preferred: readonly string[] | null | undefined): Locale | null {
  if (!preferred) {
    return null
  }

  for (const value of preferred) {
    if (isSupportedLocaleValue(value)) {
      return normalizeLocale(value)
    }
  }

  return null
}

// Resolve the UI locale to show before the user has made an explicit choice.
// An explicit, non-empty `display.language` config value always wins — mirroring
// the previous `normalizeLocale` behavior, including falling back to English for
// an unsupported configured value so a manual selection is never silently
// overridden. Only when no language is configured do we consult the system
// locale, and finally fall back to English.
export function resolveInitialLocale(
  configuredLanguage: unknown,
  systemPreferred: readonly string[] | null | undefined
): Locale {
  if (typeof configuredLanguage === 'string' && configuredLanguage.trim() !== '') {
    return normalizeLocale(configuredLanguage)
  }

  return matchPreferredLocale(systemPreferred) ?? DEFAULT_LOCALE
}
