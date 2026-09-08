import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'

import { getHermesConfigRecord, type HermesConfigRecord, saveHermesConfig } from '@/hermes'

import { TRANSLATIONS } from './catalog'
import { DEFAULT_LOCALE, localeConfigValue, normalizeLocale } from './languages'
import { setRuntimeI18nLocale } from './runtime'
import type { Locale, Translations } from './types'

export { LOCALE_META } from './languages'

export interface I18nConfigClient {
  getConfig: () => Promise<HermesConfigRecord>
  saveConfig: (config: HermesConfigRecord) => Promise<{ ok: boolean }>
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

const RTL_LOCALES = new Set<Locale>(['ar'])

// Startup config read: the renderer mounts while the backend stack from a
// `hermes update` relaunch is still settling, so the one-shot GET can reject
// in the first moments even though the persisted `display.language` is intact.
// Retry with bounded backoff instead of pinning the session to English.
const CONFIG_READ_RETRY_DELAYS_MS = [200, 400, 800, 1600, 3200]

function applyDocumentLocale(locale: Locale) {
  if (typeof document === 'undefined') {
    return
  }

  document.documentElement.lang = locale
  document.documentElement.dir = RTL_LOCALES.has(locale) ? 'rtl' : 'ltr'
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
  const localeRef = useRef(locale)

  // Set when the user picks a language through setLocale. A late config read
  // (retry resolution) must never overwrite an explicit in-session choice.
  const userLocaleRef = useRef(false)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    localeRef.current = locale
    setRuntimeI18nLocale(locale)
    applyDocumentLocale(locale)
  }, [locale])

  useEffect(() => {
    if (!configClient) {
      return
    }

    let cancelled = false
    let retryTimer: ReturnType<typeof setTimeout> | undefined

    setIsLoadingConfig(true)
    setConfigLoadError(null)

    const readConfig = async (attempt: number) => {
      try {
        const config = await configClient.getConfig()
        if (!cancelled && !userLocaleRef.current) {
          setLocaleState(normalizeLocale(getConfigDisplayLanguage(config)))
        }
      } catch (error) {
        if (cancelled) {
          return
        }

        const delay = CONFIG_READ_RETRY_DELAYS_MS[attempt]
        if (delay === undefined) {
          // Exhausted every retry: the config is genuinely unreachable. Keep
          // English usable as the deterministic fallback — unless the user has
          // already picked a language this session, which outranks a stale read.
          if (!userLocaleRef.current) {
            setConfigLoadError(toError(error))
            setLocaleState(DEFAULT_LOCALE)
          }
          return
        }

        await new Promise<void>(resolve => {
          retryTimer = setTimeout(resolve, delay)
        })
        if (!cancelled) {
          await readConfig(attempt + 1)
        }
      }
    }

    void readConfig(0).finally(() => {
      if (!cancelled) {
        setIsLoadingConfig(false)
      }
    })

    return () => {
      cancelled = true
      if (retryTimer !== undefined) {
        clearTimeout(retryTimer)
      }
    }
  }, [configClient, initialLocale])

  const setLocale = useCallback(
    async (next: Locale) => {
      const previousLocale = localeRef.current

      // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
      userLocaleRef.current = true

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
      t: TRANSLATIONS[locale]
    }),
    [configLoadError, isLoadingConfig, isSavingLocale, locale, saveError, setLocale]
  )

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n(): I18nContextValue {
  return useContext(I18nContext)
}
