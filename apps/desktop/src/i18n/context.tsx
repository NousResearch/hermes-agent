import { useStore } from '@nanostores/react'
import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'

import { $apiRequestScope, type ApiRequestScope } from '@/api/client'
import { getHermesConfigRecord, saveHermesConfigRecord } from '@/api/config'
import type { HermesConfigRecord } from '@/types/hermes'

import { TRANSLATIONS } from './catalog'
import { DEFAULT_LOCALE, localeConfigValue, normalizeLocale } from './languages'
import { setRuntimeI18nLocale } from './runtime'
import type { Locale, Translations } from './types'

export { LOCALE_META } from './languages'

export interface I18nConfigClient {
  getConfig: () => Promise<HermesConfigRecord>
  saveConfig: (config: HermesConfigRecord) => Promise<{ ok: boolean }>
}

function scopedConfigClient(scope: ApiRequestScope): I18nConfigClient {
  return {
    getConfig: () => {
      if (typeof window === 'undefined' || !window.hermesDesktop?.api) {
        return Promise.resolve({})
      }

      return getHermesConfigRecord(scope)
    },
    saveConfig: config => {
      if (typeof window === 'undefined' || !window.hermesDesktop?.api) {
        return Promise.resolve({ ok: true })
      }

      return saveHermesConfigRecord(config, scope)
    }
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

export function I18nProvider({ children, configClient, initialLocale }: I18nProviderProps) {
  const scope = useStore($apiRequestScope)
  const defaultConfigClient = useMemo(() => scopedConfigClient(scope), [scope])
  const client = configClient === undefined ? defaultConfigClient : configClient
  const [locale, setLocaleState] = useState<Locale>(() => normalizeLocale(initialLocale))
  const [isLoadingConfig, setIsLoadingConfig] = useState(false)
  const [isSavingLocale, setIsSavingLocale] = useState(false)
  const [configLoadError, setConfigLoadError] = useState<Error | null>(null)
  const [saveError, setSaveError] = useState<Error | null>(null)
  const localeRef = useRef(locale)
  // Set once the user picks a language through setLocale: a startup read that
  // resolves (or fails) after that must never overwrite an explicit choice.
  const userLocaleRef = useRef(false)
  const clientGeneration = useRef(0)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    localeRef.current = locale
    setRuntimeI18nLocale(locale)
    applyDocumentLocale(locale)
  }, [locale])

  // eslint-disable-next-line no-restricted-syntax -- lifecycle generation and user intent, not mirrored atom values
  useEffect(() => {
    clientGeneration.current += 1
    userLocaleRef.current = false
    setIsSavingLocale(false)
    setSaveError(null)

    if (!client) {
      return
    }

    let cancelled = false
    let retryTimer: ReturnType<typeof setTimeout> | null = null
    let retryCount = 0

    // The desktop races its own backend at startup: the renderer mounts before
    // the backend is ready, so the first /api/config call can time out. We keep
    // the established permanent-failure contract — a rejected config load
    // settles on English so the UI stays usable — but bounded retries recover
    // transient startup failures, applying the persisted display.language once
    // the backend comes up.
    const MAX_LOCALE_RETRIES = 10
    const LOCALE_RETRY_DELAY_MS = 3_000

    const loadLocale = () => {
      setIsLoadingConfig(true)
      setConfigLoadError(null)

      return client
        .getConfig()
        .then(config => {
          if (!cancelled && !userLocaleRef.current) {
            setLocaleState(normalizeLocale(getConfigDisplayLanguage(config)))
          }
        })
        .catch(error => {
          if (cancelled || userLocaleRef.current) {
            return
          }

          setConfigLoadError(toError(error))
          setLocaleState(DEFAULT_LOCALE)

          if (retryCount < MAX_LOCALE_RETRIES) {
            retryCount += 1
            retryTimer = setTimeout(() => {
              loadLocale()
            }, LOCALE_RETRY_DELAY_MS)
          }
        })
        .finally(() => {
          if (!cancelled) {
            setIsLoadingConfig(false)
          }
        })
    }

    loadLocale()

    return () => {
      cancelled = true
      clientGeneration.current += 1

      if (retryTimer) {
        clearTimeout(retryTimer)
      }
    }
  }, [client, initialLocale])

  const setLocale = useCallback(
    async (next: Locale) => {
      const previousLocale = localeRef.current
      const generation = clientGeneration.current

      userLocaleRef.current = true
      setSaveError(null)
      setLocaleState(next)

      if (!client) {
        return
      }

      setIsSavingLocale(true)

      try {
        const latestConfig = await client.getConfig()
        const result = await client.saveConfig(withConfigDisplayLanguage(latestConfig, next))

        if (!result.ok) {
          throw new Error('Failed to save language')
        }
      } catch (error) {
        const nextError = toError(error)

        if (clientGeneration.current === generation) {
          setLocaleState(previousLocale)
          setSaveError(nextError)
        }

        throw nextError
      } finally {
        if (clientGeneration.current === generation) {
          setIsSavingLocale(false)
        }
      }
    },
    [client]
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
