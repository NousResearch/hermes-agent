import i18n from 'i18next'
import LanguageDetector from 'i18next-browser-languagedetector'
import { initReactI18next } from 'react-i18next'

import en from '@/locales/en/translation.json'
import zhCN from '@/locales/zh-CN/translation.json'

/**
 * Languages bundled into the renderer. Add a new entry here when you ship a
 * new translation file under src/locales/<lang>/translation.json. The
 * language picker (settings/appearance) reads SUPPORTED_LANGUAGES to render
 * its options, so no other code change is needed for a new language.
 */
export const SUPPORTED_LANGUAGES = ['en', 'zh-CN'] as const
export type SupportedLanguage = (typeof SUPPORTED_LANGUAGES)[number]

/**
 * Display names for the language picker. Kept outside the translation file
 * so the picker can show the *native* name of each language regardless of
 * the currently active locale.
 */
export const LANGUAGE_LABELS: Record<SupportedLanguage, string> = {
  en: 'English',
  'zh-CN': '简体中文'
}

const LOCALE_STORAGE_KEY = 'hermes-desktop-locale-v1'

void i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: {
      en: { translation: en },
      'zh-CN': { translation: zhCN }
    },
    fallbackLng: 'en',
    supportedLngs: SUPPORTED_LANGUAGES as unknown as string[],
    // Hermes Desktop uses flat i18n keys with a `namespace:section.subsection`
    // convention (e.g. `settings:mcp.configured`). i18next's defaults treat
    // `:` as a namespace separator and `.` as a keypath separator, which would
    // try to navigate into a non-existent nested object. We want a literal
    // key match against the flat JSON, so disable both separators.
    nsSeparator: false,
    keySeparator: false,
    detection: {
      order: ['localStorage', 'navigator'],
      lookupLocalStorage: LOCALE_STORAGE_KEY,
      caches: ['localStorage']
    },
    interpolation: {
      escapeValue: false // React already escapes
    },
    react: {
      useSuspense: false
    },
    returnNull: false
  })

export default i18n
