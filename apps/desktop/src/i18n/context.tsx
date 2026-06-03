import { createContext, type ReactNode, useCallback, useContext, useMemo, useState } from 'react'

import { en } from './en'
import type { Locale, Translations } from './types'
import { zh } from './zh'

const TRANSLATIONS: Record<Locale, Translations> = {
  en,
  zh
}

// Endonyms (native names) for the language picker so users recognize their
// language regardless of the current UI language. No country flags —
// languages are not countries.
export const LOCALE_META: Record<Locale, { name: string }> = {
  en: { name: 'English' },
  zh: { name: '简体中文' }
}

const SUPPORTED_LOCALES = Object.keys(TRANSLATIONS) as Locale[]
const STORAGE_KEY = 'hermes-desktop-locale'

function isLocale(value: string): value is Locale {
  return (SUPPORTED_LOCALES as string[]).includes(value)
}

function getInitialLocale(): Locale {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY)

    if (stored && isLocale(stored)) {
      return stored
    }
  } catch {
    // localStorage unavailable (privacy mode / SSR) — fall back to English.
  }

  return 'en'
}

interface I18nContextValue {
  locale: Locale
  setLocale: (next: Locale) => void
  t: Translations
}

const I18nContext = createContext<I18nContextValue>({
  locale: 'en',
  setLocale: () => {},
  t: en
})

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(getInitialLocale)

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next)

    try {
      window.localStorage.setItem(STORAGE_KEY, next)
    } catch {
      // ignore persistence failures
    }
  }, [])

  const value = useMemo<I18nContextValue>(
    () => ({ locale, setLocale, t: TRANSLATIONS[locale] }),
    [locale, setLocale]
  )

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n(): I18nContextValue {
  return useContext(I18nContext)
}
