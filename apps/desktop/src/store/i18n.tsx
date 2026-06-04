import { createContext, useContext, useState, useCallback, type ReactNode } from 'react'

// --- Types ---
type LocaleCode = string

export const LANGUAGE_LABELS: Record<string, string> = {
  'en': 'English',
  'zh-CN': '中文（简体）',
  'zh-Hant': '中文（繁體）',
  'ja': '日本語',
  'ko': '한국어',
  'de': 'Deutsch',
  'es': 'Español',
  'fr': 'Français',
  'ar': 'العربية',
  'hi': 'हिन्दी',
  'th': 'ภาษาไทย',
  'vi': 'Tiếng Việt',
  'it': 'Italiano',
  'ru': 'Русский',
  'pt-BR': 'Português (Brasil)',
}

const STORAGE_KEY = 'hermes-desktop-locale'

// --- Auto-discover all locale files ---
// Vite's import.meta.glob eagerly imports all JSON files in the locales directory.
// To add a new language, just drop a new {code}.json file — no code changes needed!
const localeModules = import.meta.glob<Record<string, string>>('../locales/*.json', { import: 'default', eager: true })

export const SUPPORTED_LOCALES = Object.keys(localeModules)
  .map(path => path.replace('../locales/', '').replace('.json', ''))
  .sort((a, b) => a === 'en' ? -1 : b === 'en' ? 1 : 0) // English first

// Pre-load all translations
const allTranslations: Record<string, Record<string, string>> = {}
for (const [path, data] of Object.entries(localeModules)) {
  const code = path.replace('../locales/', '').replace('.json', '')
  allTranslations[code] = data ?? {}
}

// --- Module-level state ---
let _currentLocale = getInitialLocale()

export function normalizeLocale(raw: string): string {
  const lc = raw.toLowerCase()
  if (lc.startsWith('zh')) {
    if (lc === 'zh-tw' || lc === 'zh-hk' || lc === 'zh-mo') return 'zh-Hant'
    return 'zh-CN'
  }
  if (lc.startsWith('ja')) return 'ja'
  if (lc.startsWith('ko')) return 'ko'
  if (lc.startsWith('de')) return 'de'
  if (lc.startsWith('es')) return 'es'
  if (lc.startsWith('fr')) return 'fr'
  if (lc.startsWith('pt')) return 'pt-BR'
  return 'en'
}

function getInitialLocale(): string {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved && SUPPORTED_LOCALES.includes(saved)) return saved
  } catch {}
  return normalizeLocale(typeof navigator !== 'undefined' ? navigator.language : 'en')
}

// --- React Context ---
interface I18nContextValue {
  locale: string
  t: (key: string, params?: Record<string, unknown>) => string
  setLocale: (locale: string) => void
  availableLocales: string[]
}

const I18nContext = createContext<I18nContextValue | null>(null)

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<string>(_currentLocale)

  const setLocale = useCallback((newLocale: string) => {
    if (!SUPPORTED_LOCALES.includes(newLocale)) return
    _currentLocale = newLocale
    setLocaleState(newLocale)
    notifyListeners()
    try { localStorage.setItem(STORAGE_KEY, newLocale) } catch {}
  }, [])

  const t = useCallback((key: string, params?: Record<string, unknown>): string => {
    const translations = allTranslations[_currentLocale]
    const fallback = allTranslations['en']
    let value = translations?.[key] ?? fallback?.[key] ?? key
    if (params) {
      value = Object.entries(params).reduce(
        (str, [k, v]) => str.replace(new RegExp(`\\{${k}\\}`, 'g'), String(v)),
        value,
      )
    }
    return value
  }, [locale])

  return (
    <I18nContext.Provider value={{ locale, t, setLocale, availableLocales: SUPPORTED_LOCALES }}>
      {children}
    </I18nContext.Provider>
  )
}

export function useTranslation(): I18nContextValue {
  const ctx = useContext(I18nContext)
  if (!ctx) throw new Error('useTranslation must be used within I18nProvider')
  return ctx
}

// --- Locale change listeners (for components using standalone t()) ---
let _listeners: Set<() => void> = new Set()

export function onLocaleChange(fn: () => void): () => void {
  _listeners.add(fn)
  return () => { _listeners.delete(fn) }
}

function notifyListeners() {
  _listeners.forEach(fn => fn())
}

// Hook: forces re-render when locale changes
// Use in components that import t() directly instead of useTranslation()
export { useLocaleSync } from './use-locale-sync'

// --- Standalone t() for non-React code ---
export function t(key: string, params?: Record<string, unknown>): string {
  const translations = allTranslations[_currentLocale]
  const fallback = allTranslations['en']
  let value = translations?.[key] ?? fallback?.[key] ?? key
  if (params) {
    value = Object.entries(params).reduce(
      (str, [k, v]) => str.replace(new RegExp(`\\{${k}\\}`, 'g'), String(v)),
      value,
    )
  }
  return value
}
