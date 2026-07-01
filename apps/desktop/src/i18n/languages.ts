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
  },
  {
    id: 'ko',
    name: '한국어',
    englishName: 'Korean',
    configValue: 'ko'
  },
  {
    id: 'de',
    name: 'Deutsch',
    englishName: 'German',
    configValue: 'de'
  },
  {
    id: 'es',
    name: 'Español',
    englishName: 'Spanish',
    configValue: 'es'
  },
  {
    id: 'fr',
    name: 'Français',
    englishName: 'French',
    configValue: 'fr'
  },
  {
    id: 'pt-br',
    name: 'Português (Brasil)',
    englishName: 'Portuguese (Brazil)',
    configValue: 'pt-br'
  },
  {
    id: 'ar',
    name: 'العربية',
    englishName: 'Arabic',
    configValue: 'ar'
  },
  {
    id: 'hi',
    name: 'हिन्दी',
    englishName: 'Hindi',
    configValue: 'hi'
  },
  {
    id: 'th',
    name: 'ภาษาไทย',
    englishName: 'Thai',
    configValue: 'th'
  },
  {
    id: 'vi',
    name: 'Tiếng Việt',
    englishName: 'Vietnamese',
    configValue: 'vi'
  },
  {
    id: 'it',
    name: 'Italiano',
    englishName: 'Italian',
    configValue: 'it'
  },
  {
    id: 'ru',
    name: 'Русский',
    englishName: 'Russian',
    configValue: 'ru'
  },
] as const satisfies readonly { configValue: string; englishName: string; id: Locale; name: string }[]

// `name` is the endonym (native name) shown in the picker so users recognize
// their language regardless of the current UI language. No country flags:
// languages are not countries. `englishName` is search-only (not shown) so an
// English speaker can type "japanese"/"traditional" to filter the list.
export const LOCALE_META: Record<Locale, { name: string; englishName: string }> = Object.fromEntries(
  LOCALE_OPTIONS.map(locale => [locale.id, { name: locale.name, englishName: locale.englishName }])
) as Record<Locale, { name: string; englishName: string }>

const LOCALE_ALIASES: Record<string, Locale> = {
  'ar': 'ar',
  'ar-ae': 'ar',
  'ar-eg': 'ar',
  'ar-sa': 'ar',
  'ar_sa': 'ar',
  'de': 'de',
  'de-at': 'de',
  'de-ch': 'de',
  'de-de': 'de',
  'de_de': 'de',
  'en': 'en',
  'en-us': 'en',
  'en_us': 'en',
  'es': 'es',
  'es-ar': 'es',
  'es-es': 'es',
  'es-mx': 'es',
  'es_es': 'es',
  'fr': 'fr',
  'fr-be': 'fr',
  'fr-ca': 'fr',
  'fr-ch': 'fr',
  'fr-fr': 'fr',
  'fr_fr': 'fr',
  'hi': 'hi',
  'hi-in': 'hi',
  'hi_in': 'hi',
  'it': 'it',
  'it-ch': 'it',
  'it-it': 'it',
  'it_it': 'it',
  'ja': 'ja',
  'ja-jp': 'ja',
  'ja_jp': 'ja',
  'ko': 'ko',
  'ko-kr': 'ko',
  'ko_kr': 'ko',
  'pt-br': 'pt-br',
  'pt-pt': 'pt-br',
  'pt_br': 'pt-br',
  'pt_pt': 'pt-br',
  'ru': 'ru',
  'ru-ru': 'ru',
  'ru_ru': 'ru',
  'th': 'th',
  'th-th': 'th',
  'th_th': 'th',
  'vi': 'vi',
  'vi-vn': 'vi',
  'vi_vn': 'vi',
  'zh': 'zh',
  'zh-cn': 'zh',
  'zh-hans': 'zh',
  'zh-hans-cn': 'zh',
  'zh-hant': 'zh-hant',
  'zh-hant-hk': 'zh-hant',
  'zh-hant-tw': 'zh-hant',
  'zh-hk': 'zh-hant',
  'zh-mo': 'zh-hant',
  'zh-tw': 'zh-hant',
  'zh_cn': 'zh',
  'zh_hans': 'zh',
  'zh_hans_cn': 'zh',
  'zh_hant': 'zh-hant',
  'zh_hant_hk': 'zh-hant',
  'zh_hant_tw': 'zh-hant',
  'zh_hk': 'zh-hant',
  'zh_mo': 'zh-hant',
  'zh_tw': 'zh-hant',
}

export function isLocale(value: unknown): value is Locale {
  return typeof value === 'string' && LOCALE_OPTIONS.some(locale => locale.id === value)
}

export function normalizeLocale(value: unknown): Locale {
  if (typeof value !== 'string') {
    return DEFAULT_LOCALE
  }

  return LOCALE_ALIASES[value.trim().toLowerCase()] ?? DEFAULT_LOCALE
}

export function isSupportedLocaleValue(value: unknown): boolean {
  return typeof value === 'string' && LOCALE_ALIASES[value.trim().toLowerCase()] != null
}

export function localeConfigValue(locale: Locale): string {
  return LOCALE_OPTIONS.find(item => item.id === locale)?.configValue ?? DEFAULT_LOCALE
}
