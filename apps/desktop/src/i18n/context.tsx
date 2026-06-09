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

const CONFIG_LOAD_MAX_ATTEMPTS = 5
const CONFIG_LOAD_RETRY_BASE_DELAY_MS = 250

function getConfigLoadRetryDelayMs(attempt: number): number {
  return CONFIG_LOAD_RETRY_BASE_DELAY_MS * 2 ** attempt
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

  useEffect(() => {
    localeRef.current = locale
    setRuntimeI18nLocale(locale)
  }, [locale])

  useEffect(() => {
    if (!configClient) {
      return
    }

    let cancelled = false
    let retryTimer: ReturnType<typeof globalThis.setTimeout> | null = null
    let resolveRetryWait: (() => void) | null = null

    const finishRetryWait = () => {
      const resolve = resolveRetryWait
      resolveRetryWait = null
      resolve?.()
    }

    const waitForRetry = (ms: number) =>
      new Promise<void>(resolve => {
        resolveRetryWait = resolve
        retryTimer = globalThis.setTimeout(() => {
          retryTimer = null
          finishRetryWait()
        }, ms)
      })

    setIsLoadingConfig(true)
    setConfigLoadError(null)

    void (async () => {
      let lastError: Error | null = null

      for (let attempt = 0; attempt < CONFIG_LOAD_MAX_ATTEMPTS; attempt += 1) {
        try {
          const config = await configClient.getConfig()

          if (!cancelled) {
            setLocaleState(normalizeLocale(getConfigDisplayLanguage(config)))
          }

          return
        } catch (error) {
          lastError = toError(error)

          if (cancelled) {
            return
          }

          if (attempt < CONFIG_LOAD_MAX_ATTEMPTS - 1) {
            await waitForRetry(getConfigLoadRetryDelayMs(attempt))

            if (cancelled) {
              return
            }
          }
        }
      }

      if (!cancelled && lastError) {
        setConfigLoadError(lastError)
        setLocaleState(DEFAULT_LOCALE)
      }
    })().finally(() => {
      if (!cancelled) {
        setIsLoadingConfig(false)
      }
    })

    return () => {
      cancelled = true

      if (retryTimer !== null) {
        globalThis.clearTimeout(retryTimer)
        retryTimer = null
      }

      finishRetryWait()
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
      t: TRANSLATIONS[locale]
    }),
    [configLoadError, isLoadingConfig, isSavingLocale, locale, saveError, setLocale]
  )

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n(): I18nContextValue {
  return useContext(I18nContext)
}
