import { TRANSLATIONS } from './catalog'
import { DEFAULT_LOCALE } from './languages'
import type { Locale, Translations } from './types'

let runtimeLocale: Locale = DEFAULT_LOCALE

// Per-locale catalogs with user overrides merged in. Populated by the
// I18nProvider once override files are read from disk; absent locales fall back
// to the bundled TRANSLATIONS. Keeps the imperative translateNow() path in sync
// with the React context.
const runtimeOverrides: Partial<Record<Locale, Translations>> = {}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function catalogFor(locale: Locale): Translations {
  return runtimeOverrides[locale] ?? TRANSLATIONS[locale]
}

function resolvePath(catalog: Translations, key: string): unknown {
  return key.split('.').reduce<unknown>((current, part) => {
    if (!isRecord(current)) {
      return undefined
    }

    return current[part]
  }, catalog)
}

function renderTranslation(value: unknown, args: unknown[]): string | null {
  if (typeof value === 'string') {
    return value
  }

  if (typeof value === 'function') {
    return (value as (...args: unknown[]) => string)(...args)
  }

  return null
}

export function setRuntimeI18nLocale(locale: Locale) {
  runtimeLocale = locale
}

/**
 * Register (or clear) the override-merged catalog for a locale. Pass null to
 * drop back to the bundled catalog. Called by the I18nProvider after reading
 * the user's override file.
 */
export function setRuntimeLocaleOverride(locale: Locale, catalog: Translations | null) {
  if (catalog) {
    runtimeOverrides[locale] = catalog
  } else {
    delete runtimeOverrides[locale]
  }
}

export function translateNow(key: string, ...args: unknown[]): string {
  const active = renderTranslation(resolvePath(catalogFor(runtimeLocale), key), args)

  if (active !== null) {
    return active
  }

  if (runtimeLocale !== DEFAULT_LOCALE) {
    const fallback = renderTranslation(resolvePath(catalogFor(DEFAULT_LOCALE), key), args)

    if (fallback !== null) {
      return fallback
    }
  }

  return key
}
