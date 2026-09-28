import { type Locale, LOCALE_METADATA, localeDirection, LOCALES } from './locale-registry.js'

// Locale scaffolding shared by the desktop and web i18n layers. Generic over the
// translation catalog type: each app supplies its own `Translations`/`en` and
// wraps `mergeTranslations` in a one-line `defineLocale`.

/** Partial-locale shape: every key optional, but functions/arrays are atomic and
 *  unknown keys still fail the type-check. */
export type TranslationOverride<T> = T extends (...args: never[]) => string
  ? T
  : T extends readonly unknown[]
    ? T
    : T extends string
      ? string
      : T extends object
        ? { [K in keyof T]?: TranslationOverride<T[K]> }
        : T

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Deep-merge a partial locale over the English base: nested records recurse,
 *  everything else (strings, functions, arrays) is replaced wholesale. */
export function mergeTranslations<T>(base: T, overrides: TranslationOverride<T> | undefined): T {
  if (!isRecord(base) || !isRecord(overrides)) {
    return (overrides ?? base) as T
  }

  const result: Record<string, unknown> = { ...base }

  for (const [key, value] of Object.entries(overrides)) {
    if (value === undefined) {
      continue
    }

    const baseValue = result[key]
    result[key] = isRecord(baseValue) && isRecord(value) ? mergeTranslations(baseValue, value) : value
  }

  return result as T
}

// Endonyms (native names) for the language pickers so users recognize their
// language regardless of the current UI language. No country flags: languages
// are not countries (English ≠ GB, Portuguese ≠ PT, Chinese variants ≠ any
// single jurisdiction). Desktop supports a subset of these ids; web all of them.
export const LOCALE_ENDONYMS = Object.fromEntries(
  LOCALES.map(locale => [locale, LOCALE_METADATA[locale].name])
) as Record<Locale, string>

export type EndonymLocale = Locale

export const RTL_LOCALES: ReadonlySet<string> = new Set(LOCALES.filter(locale => localeDirection(locale) === 'rtl'))
