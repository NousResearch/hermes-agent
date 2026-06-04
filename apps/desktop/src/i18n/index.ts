import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import LanguageDetector from 'i18next-browser-languagedetector'

import en from './locales/en.json'
import zh from './locales/zh.json'

const STORED_KEY = 'hermes-desktop-locale'

function getPersistedLocale(): string | undefined {
  try {
    return localStorage.getItem(STORED_KEY) ?? undefined
  } catch {
    return undefined
  }
}

function persistLocale(locale: string): void {
  try {
    localStorage.setItem(STORED_KEY, locale)
  } catch {}
}

const persistedLocale = getPersistedLocale()

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: {
      en: { translation: en },
      zh: { translation: zh }
    },
    lng: persistedLocale ?? undefined,
    fallbackLng: 'en',
    supportedLngs: ['en', 'zh'],
    interpolation: {
      escapeValue: false
    },
    detection: {
      order: ['navigator'],
      caches: []
    }
  })

i18n.on('languageChanged', persistLocale)

export default i18n
