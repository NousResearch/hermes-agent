import { LOCALE_ENDONYMS, RTL_LOCALES } from '@hermes/shared/i18n'

import { normalize } from '@/lib/text'

import { isBundledLocale } from './catalog'
import {
  type AppLocaleSource,
  isRegisteredLocale,
  normalizeLocaleId,
  registeredLocaleIds,
  registeredLocaleMeta
} from './registry'
import type { BundledLocale, Locale } from './types'

export const DEFAULT_LOCALE: BundledLocale = 'en'

export const LOCALE_OPTIONS = [
  {
    id: 'en',
    name: LOCALE_ENDONYMS.en,
    englishName: 'English',
    configValue: 'en'
  },
  {
    id: 'zh',
    name: LOCALE_ENDONYMS.zh,
    englishName: 'Simplified Chinese',
    configValue: 'zh'
  },
  {
    id: 'zh-hant',
    name: LOCALE_ENDONYMS['zh-hant'],
    englishName: 'Traditional Chinese',
    configValue: 'zh-hant'
  },
  {
    id: 'ja',
    name: LOCALE_ENDONYMS.ja,
    englishName: 'Japanese',
    configValue: 'ja'
  },
  {
    id: 'ar',
    name: LOCALE_ENDONYMS.ar,
    englishName: 'Arabic',
    configValue: 'ar'
  },
  {
    id: 'ru',
    name: LOCALE_ENDONYMS.ru,
    englishName: 'Russian',
    configValue: 'ru'
  },
  {
    id: 'fr',
    name: LOCALE_ENDONYMS.fr,
    englishName: 'French',
    configValue: 'fr'
  },
  {
    id: 'de',
    name: LOCALE_ENDONYMS.de,
    englishName: 'German',
    configValue: 'de'
  },
  {
    id: 'es',
    name: LOCALE_ENDONYMS.es,
    englishName: 'Spanish',
    configValue: 'es'
  },
  {
    id: 'id',
    name: LOCALE_ENDONYMS.id,
    englishName: 'Indonesian',
    configValue: 'id'
  }
] as const satisfies readonly { configValue: string; englishName: string; id: BundledLocale; name: string }[]

export interface LanguageOption {
  id: Locale
  /** Native name shown in the picker (`Polski`). */
  endonym: string
  /** Search-only English name; absent for a registered language that gave none. */
  englishName?: string
  rtl: boolean
  source: 'bundled' | AppLocaleSource
}

const bundledOption = new Map<string, (typeof LOCALE_OPTIONS)[number]>(
  LOCALE_OPTIONS.map(option => [option.id, option])
)

/** Endonym / search name / direction for one id: registry (last word wins)
 *  → bundled picker table → shared endonyms → the id itself. */
export function localeMeta(locale: Locale): Pick<LanguageOption, 'endonym' | 'englishName' | 'rtl'> {
  const registered = registeredLocaleMeta(locale).reverse()
  const bundled = bundledOption.get(locale)
  const rtlEntry = registered.find(entry => entry.rtl !== undefined)

  return {
    endonym:
      registered.find(entry => entry.endonym)?.endonym ??
      bundled?.name ??
      (LOCALE_ENDONYMS as Record<string, string>)[locale] ??
      locale,
    englishName: registered.find(entry => entry.englishName)?.englishName ?? bundled?.englishName,
    rtl: rtlEntry ? Boolean(rtlEntry.rtl) : RTL_LOCALES.has(locale)
  }
}

export function isRtlLocale(locale: Locale): boolean {
  return localeMeta(locale).rtl
}

// The endonym (native name) is what the picker shows so users recognize their
// language regardless of the current UI language. No country flags: languages
// are not countries. `englishName` is search-only (not shown) so an English
// speaker can type "japanese"/"traditional" to filter the list.
/** Every selectable language: bundled ∪ registered (plugin packs, backend
 *  `i18n.languages`). Bundled keep their curated picker order; registered-only
 *  ids follow, sorted. */
export function languageOptions(): LanguageOption[] {
  const options: LanguageOption[] = LOCALE_OPTIONS.map(option => ({
    id: option.id,
    ...localeMeta(option.id),
    source: 'bundled'
  }))

  for (const id of registeredLocaleIds()
    .filter(id => !isBundledLocale(id))
    .sort()) {
    options.push({ id, ...localeMeta(id), source: registeredLocaleMeta(id)[0]?.source ?? 'app' })
  }

  return options
}

const LOCALE_ALIASES: Record<string, BundledLocale> = {
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
  ja_jp: 'ja',
  ar: 'ar',
  'ar-sa': 'ar',
  ar_sa: 'ar',
  'ar-ae': 'ar',
  ar_ae: 'ar',
  'ar-eg': 'ar',
  ar_eg: 'ar',
  arabic: 'ar',
  العربية: 'ar',
  ru: 'ru',
  'ru-ru': 'ru',
  ru_ru: 'ru',
  'ru-by': 'ru',
  'ru-kz': 'ru',
  russian: 'ru',
  'russian-russian': 'ru',
  русский: 'ru',
  руский: 'ru',
  fr: 'fr',
  'fr-fr': 'fr',
  fr_fr: 'fr',
  'fr-be': 'fr',
  fr_be: 'fr',
  'fr-ca': 'fr',
  fr_ca: 'fr',
  'fr-ch': 'fr',
  fr_ch: 'fr',
  french: 'fr',
  français: 'fr',
  francais: 'fr',
  de: 'de',
  'de-de': 'de',
  de_de: 'de',
  'de-at': 'de',
  de_at: 'de',
  'de-ch': 'de',
  de_ch: 'de',
  german: 'de',
  deutsch: 'de',
  es: 'es',
  'es-es': 'es',
  es_es: 'es',
  'es-mx': 'es',
  es_mx: 'es',
  'es-ar': 'es',
  es_ar: 'es',
  'es-419': 'es',
  es_419: 'es',
  spanish: 'es',
  español: 'es',
  espanol: 'es',
  id: 'id',
  'id-id': 'id',
  id_id: 'id',
  indonesian: 'id',
  bahasa: 'id',
  'bahasa-indonesia': 'id',
  bahasa_indonesia: 'id'
}

/** A language the app can render right now: bundled or registered. Aliases
 *  (`zh-CN`) are not ids — see `isSupportedLocaleValue`. */
export function isLocale(value: unknown): value is Locale {
  return isBundledLocale(value) || isRegisteredLocale(value)
}

/** Alias table first (`zh-TW` → `zh-hant`), then any registered id as-is
 *  (lowercased, `_` → `-`), else English. */
export function normalizeLocale(value: unknown): Locale {
  if (typeof value !== 'string') {
    return DEFAULT_LOCALE
  }

  const alias = LOCALE_ALIASES[normalize(value)]

  if (alias) {
    return alias
  }

  const id = normalizeLocaleId(value)

  return isRegisteredLocale(id) ? id : DEFAULT_LOCALE
}

export function isSupportedLocaleValue(value: unknown): boolean {
  return (
    typeof value === 'string' &&
    (LOCALE_ALIASES[normalize(value)] != null || isRegisteredLocale(normalizeLocaleId(value)))
  )
}

/** OS tags can include regions absent from the picker aliases, such as ru-UA. */
export function osPreferredLocale(tag: string | null | undefined): Locale | null {
  if (!tag) {
    return null
  }

  const exact = LOCALE_ALIASES[normalize(tag)]

  if (exact) {
    return exact
  }

  const base = tag.split(/[-_]/)[0]

  return (base && LOCALE_ALIASES[normalize(base)]) || null
}

/** An explicit choice must win even when it differs from the OS language. */
export function resolveInitialLocale(saved: string | null | undefined, osLocale: string | null | undefined): Locale {
  if (isSupportedLocaleValue(saved)) {
    return normalizeLocale(saved)
  }

  return osPreferredLocale(osLocale) ?? DEFAULT_LOCALE
}

/** The `display.language` value for a picker choice — the bundled option's
 *  config value, or the registered id itself. */
export function localeConfigValue(locale: Locale): string {
  return bundledOption.get(locale)?.configValue ?? (isRegisteredLocale(locale) ? locale : DEFAULT_LOCALE)
}
