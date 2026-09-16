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
export const LOCALE_ENDONYMS = {
  af: 'Afrikaans',
  ar: 'العربية',
  de: 'Deutsch',
  en: 'English',
  es: 'Español',
  fr: 'Français',
  ga: 'Gaeilge',
  hu: 'Magyar',
  it: 'Italiano',
  ja: '日本語',
  ko: '한국어',
  pt: 'Português',
  ru: 'Русский',
  tr: 'Türkçe',
  uk: 'Українська',
  zh: '简体中文',
  'zh-hant': '繁體中文'
} as const satisfies Record<string, string>

export type EndonymLocale = keyof typeof LOCALE_ENDONYMS

/** Locales whose script flows right-to-left; drives `<html dir>` so Tailwind's
 *  logical utilities (ms-/me-, ps-/pe-) flip. */
export const RTL_LOCALES: ReadonlySet<string> = new Set<EndonymLocale>(['ar'])

/** Mirror the active locale onto `<html lang dir>`. No-op without a document (SSR, tests). */
export function applyDocumentLocale(locale: string): void {
  if (typeof document === 'undefined') {
    return
  }

  document.documentElement.lang = locale
  document.documentElement.dir = RTL_LOCALES.has(locale) ? 'rtl' : 'ltr'
}
