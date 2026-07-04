import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'

import { getHermesConfigRecord, type HermesConfigRecord, saveHermesConfig } from '@/hermes'

import { TRANSLATIONS } from './catalog'
import { DEFAULT_LOCALE, localeConfigValue, normalizeLocale } from './languages'
import { applyLocaleOverrides } from './overrides'
import { setRuntimeI18nLocale, setRuntimeLocaleOverride } from './runtime'
import type { Locale, Translations } from './types'

export { LOCALE_META } from './languages'

export interface I18nConfigClient {
  getConfig: () => Promise<HermesConfigRecord>
  saveConfig: (config: HermesConfigRecord) => Promise<{ ok: boolean }>
  // Optional: user-authored locale overrides read from disk (outside the app
  // bundle) so they survive updates. Absent/unsupported clients simply skip the
  // override layer and use the bundled catalog.
  getLocaleOverrides?: (locale: Locale) => Promise<unknown>
}

const defaultConfigClient: I18nConfigClient = {
  getConfig: () => {
    if (typeof window === 'undefined' || !window.hermesDesktop?.api) {
      return Promise.resolve({})
    }

    return getHermesConfigRecord()
  },
  saveConfig: config => {
    if (typeof window === 'undefined' || !window.hermesDesktop?.api) {
      return Promise.resolve({ ok: true })
    }

    return saveHermesConfig(config)
  },
  getLocaleOverrides: locale => {
    if (typeof window === 'undefined' || !window.hermesDesktop?.getLocaleOverrides) {
      return Promise.resolve(null)
    }

    return window.hermesDesktop.getLocaleOverrides(locale)
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function getConfigDisplayLanguage(config: HermesConfigRecord): unknown {
  return isRecord(config.display) ? config.display.language : undefined
}

export function withConfigDisplayLanguage(config: HermesConfigRecord, locale: Locale): HermesConfigRecord {
  const display = isRecord(config.display) ? config.display : {}

  return {
    ...config,
    display: {
      ...display,
      language: localeConfigValue(locale)
    }
  }
}

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error))
}

export interface I18nContextValue {
  configLoadError: Error | null
  isLoadingConfig: boolean
  isSavingLocale: boolean
  locale: Locale
  saveError: Error | null
  setLocale: (next: Locale) => Promise<void>
  t: Translations
}

const I18nContext = createContext<I18nContextValue>({
  configLoadError: null,
  isLoadingConfig: false,
  isSavingLocale: false,
  locale: DEFAULT_LOCALE,
  saveError: null,
  setLocale: async () => {},
  t: TRANSLATIONS[DEFAULT_LOCALE]
})

export interface I18nProviderProps {
  children: ReactNode
  configClient?: I18nConfigClient | null
  initialLocale?: unknown
}

export function I18nProvider({ children, configClient = defaultConfigClient, initialLocale }: I18nProviderProps) {
  const [locale, setLocaleState] = useState<Locale>(() => normalizeLocale(initialLocale))
  const [isLoadingConfig, setIsLoadingConfig] = useState(false)
  const [isSavingLocale, setIsSavingLocale] = useState(false)
  const [configLoadError, setConfigLoadError] = useState<Error | null>(null)
  const [saveError, setSaveError] = useState<Error | null>(null)
  // The active catalog with user overrides merged in. Null until (and unless)
  // an override file is found for the current locale; otherwise we use the
  // bundled TRANSLATIONS[locale] directly.
  const [overrideCatalog, setOverrideCatalog] = useState<{ locale: Locale; catalog: Translations } | null>(null)
  const localeRef = useRef(locale)

  useEffect(() => {
    localeRef.current = locale
    setRuntimeI18nLocale(locale)
  }, [locale])

  // Load user locale overrides for the active locale and merge them on top of
  // the bundled catalog. Overrides live outside the bundle, so they survive
  // updates. Failures are swallowed — the bundled catalog is always usable.
  useEffect(() => {
    if (!configClient?.getLocaleOverrides) {
      return
    }

    let cancelled = false

    configClient
      .getLocaleOverrides(locale)
      .then(data => {
        if (cancelled) {
          return
        }

        const merged = applyLocaleOverrides(TRANSLATIONS[locale], data)

        if (merged === TRANSLATIONS[locale]) {
          // No effective override — clear any stale merged catalog.
          setOverrideCatalog(prev => (prev?.locale === locale ? null : prev))
          setRuntimeLocaleOverride(locale, null)
          return
        }

        setOverrideCatalog({ locale, catalog: merged })
        setRuntimeLocaleOverride(locale, merged)
      })
      .catch(() => {
        // Ignore — a missing or unreadable override just means no override.
      })

    return () => {
      cancelled = true
    }
  }, [configClient, locale])

  useEffect(() => {
    if (!configClient) {
      return
    }

    let cancelled = false

    setIsLoadingConfig(true)
    setConfigLoadError(null)

    configClient
      .getConfig()
      .then(config => {
        if (!cancelled) {
          setLocaleState(normalizeLocale(getConfigDisplayLanguage(config)))
        }
      })
      .catch(error => {
        if (!cancelled) {
          setConfigLoadError(toError(error))
          setLocaleState(DEFAULT_LOCALE)
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingConfig(false)
        }
      })

    return () => {
      cancelled = true
    }
  }, [configClient, initialLocale])

  const setLocale = useCallback(
    async (next: Locale) => {
      const previousLocale = localeRef.current

      setSaveError(null)
      setLocaleState(next)

      if (!configClient) {
        return
      }

      setIsSavingLocale(true)

      try {
        const latestConfig = await configClient.getConfig()
        const result = await configClient.saveConfig(withConfigDisplayLanguage(latestConfig, next))

        if (!result.ok) {
          throw new Error('Failed to save language')
        }
      } catch (error) {
        const nextError = toError(error)

        setLocaleState(previousLocale)
        setSaveError(nextError)

        throw nextError
      } finally {
        setIsSavingLocale(false)
      }
    },
    [configClient]
  )

  const value = useMemo<I18nContextValue>(
    () => ({
      configLoadError,
      isLoadingConfig,
      isSavingLocale,
      locale,
      saveError,
      setLocale,
      t: overrideCatalog?.locale === locale ? overrideCatalog.catalog : TRANSLATIONS[locale]
    }),
    [configLoadError, isLoadingConfig, isSavingLocale, locale, overrideCatalog, saveError, setLocale]
  )

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n(): I18nContextValue {
  return useContext(I18nContext)
}
