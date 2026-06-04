import { createContext, useContext, useState, useCallback, type ReactNode } from 'react'

// --- Types ---
export type Locale = 'en' | 'zh-CN' | 'ja' | 'ko' | 'de' | 'es' | 'fr'

export const SUPPORTED_LOCALES: Locale[] = ['en', 'zh-CN', 'ja', 'ko', 'de', 'es', 'fr']

export const LANGUAGE_LABELS: Record<Locale, string> = {
  'en': 'English',
  'zh-CN': '中文（简体）',
  'ja': '日本語',
  'ko': '한국어',
  'de': 'Deutsch',
  'es': 'Español',
  'fr': 'Français',
}

const STORAGE_KEY = 'hermes-desktop-locale'

// --- Auto-discover locale modules ---
// To add a new language, just add the Locale type above + a new JSON file in src/locales/
// No other code changes needed — the import map below is the single registration point.
const localeLoaders: Record<string, () => Promise<Record<string, string>>> = {
  'en':  () => import('../locales/en.json').then(m => (m.default ?? m) as Record<string, string>),
  'zh-CN': () => import('../locales/zh-CN.json').then(m => (m.default ?? m) as Record<string, string>),
  'ja': () => import('../locales/ja.json').then(m => (m.default ?? m) as Record<string, string>),
  'ko': () => import('../locales/ko.json').then(m => (m.default ?? m) as Record<string, string>),
  'de': () => import('../locales/de.json').then(m => (m.default ?? m) as Record<string, string>),
  'es': () => import('../locales/es.json').then(m => (m.default ?? m) as Record<string, string>),
  'fr': () => import('../locales/fr.json').then(m => (m.default ?? m) as Record<string, string>),
}

// --- Module-level cache ---
const allTranslations: Record<string, Record<string, string>> = {}
let _currentLocale: Locale = getInitialLocale()

// --- Helpers ---
function normalizeLocale(raw: string): Locale {
  const lc = raw.toLowerCase()
  if (lc.startsWith('zh')) return 'zh-CN'
  if (lc.startsWith('ja')) return 'ja'
  if (lc.startsWith('ko')) return 'ko'
  if (lc.startsWith('de')) return 'de'
  if (lc.startsWith('es')) return 'es'
  if (lc.startsWith('fr')) return 'fr'
  return 'en'
}

function getInitialLocale(): Locale {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved && (SUPPORTED_LOCALES as string[]).includes(saved)) {
      return saved as Locale
    }
  } catch {}
  return normalizeLocale(typeof navigator !== 'undefined' ? navigator.language : 'en')
}

async function ensureLocale(locale: Locale): Promise<void> {
  if (allTranslations[locale]) return
  const loader = localeLoaders[locale]
  if (loader) {
    try {
      allTranslations[locale] = await loader()
    } catch {
      // Fall through — English fallback below
    }
  }
  // Fallback: load English if available
  if (!allTranslations[locale] && locale !== 'en') {
    if (!allTranslations['en']) {
      const enLoader = localeLoaders['en']
      if (enLoader) allTranslations['en'] = await enLoader()
    }
    allTranslations[locale] = allTranslations['en'] ?? {}
  }
}

// --- React Context ---
interface I18nContextValue {
  locale: Locale
  t: (key: string, params?: Record<string, unknown>) => string
  setLocale: (locale: Locale) => void
  availableLocales: Locale[]
}

const I18nContext = createContext<I18nContextValue | null>(null)

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(_currentLocale)

  const setLocale = useCallback((newLocale: Locale) => {
    if (!(SUPPORTED_LOCALES as string[]).includes(newLocale)) return
    _currentLocale = newLocale
    setLocaleState(newLocale)
    try { localStorage.setItem(STORAGE_KEY, newLocale) } catch {}
    // Preload translations
    ensureLocale(newLocale)
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
      <div key={locale}>{children}</div>
    </I18nContext.Provider>
  )
}

export function useTranslation(): I18nContextValue {
  const ctx = useContext(I18nContext)
  if (!ctx) throw new Error('useTranslation must be used within I18nProvider')
  return ctx
}

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


// --- Provider description translation ---
// Maps known backend description strings to i18n keys.
// When a provider description matches, use the translated version.
const PROVIDER_DESC_KEYS: Record<string, string> = {
  "Nous Portal base URL override": "provider.desc.nousPortal",
  "OpenRouter API key (for vision, web scraping helpers, and MoA)": "provider.desc.openRouter",
  "Google AI Studio API key (also recognized as GEMINI_API_KEY)": "provider.desc.google",
  "Google AI Studio API key (alias for GOOGLE_API_KEY)": "provider.desc.gemini",
  "Anthropic API key": "provider.desc.anthropic",
  "DeepSeek API key": "provider.desc.deepseek",
  "xAI API key": "provider.desc.xai",
  "Hugging Face API token": "provider.desc.huggingface",
  "Kimi API key (Moonshot)": "provider.desc.kimi",
  "MiniMax API key": "provider.desc.minimax",
  "MiniMax China API key": "provider.desc.minimaxCn",
  "DashScope API key (Alibaba Qwen)": "provider.desc.dashscope",
  "GLM API key (Z.AI)": "provider.desc.glm",
  "NVIDIA NIM API key (build.nvidia.com or local NIM endpoint)": "provider.desc.nvidia",
  "OpenCode Go API key": "provider.desc.opencodeGo",
  "OpenCode Zen API key": "provider.desc.opencodeZen",
  "Xiaomi MiMo API key": "provider.desc.xiaomi",
  "Kilo Code API key": "provider.desc.kiloCode",
  "AI Gateway API key (Vercel AI Gateway)": "provider.desc.aiGateway",
  "BROWSERBASE_API_KEY": "provider.desc.browserbase",
  "ElevenLabs API key": "provider.desc.elevenLabs",
  "OpenAI API key (for voice processing via proxy)": "provider.desc.openai",
  "Mistral API key (for STT/TTS)": "provider.desc.mistral",
  "Groq API key (for STT)": "provider.desc.groq",
  "TAVILY_API_KEY": "provider.desc.tavily",
  "FIRECRAWL_API_KEY": "provider.desc.firecrawl",
  "FAL_KEY": "provider.desc.fal",
}

export function translateProviderDesc(desc: string): string {
  const key = PROVIDER_DESC_KEYS[desc]
  if (key) {
    const translations = allTranslations[_currentLocale]
    const fallback = allTranslations['en']
    return translations?.[key] ?? fallback?.[key] ?? desc
  }
  return desc
}

// --- Bootstrap: preload English + saved locale ---
;(async () => {
  await ensureLocale('en')
  const saved = _currentLocale
  if (saved !== 'en') await ensureLocale(saved)
})()
