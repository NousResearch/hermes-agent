import { applyDocumentLocale, isRecord } from '@hermes/shared/i18n'
import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'

import { getHermesConfigRecord, type HermesConfigRecord, saveHermesConfig } from '@/hermes'

import { TRANSLATIONS } from './catalog'
import {
  DEFAULT_LOCALE,
  isSupportedLocaleValue,
  localeConfigValue,
  normalizeLocale,
  resolveInitialLocale
} from './languages'
import { setRuntimeI18nLocale } from './runtime'
import type { Locale, Translations } from './types'

export { LOCALE_META } from './languages'

export interface I18nConfigClient {
  getConfig: (profile?: string | null) => Promise<HermesConfigRecord>
  saveConfig: (config: HermesConfigRecord, profile?: string | null) => Promise<{ ok: boolean }>
}

const defaultConfigClient: I18nConfigClient = {
  getConfig: profile => {
    if (typeof window === 'undefined' || !window.hermesDesktop?.api) {
      return Promise.resolve({})
    }

    // Merged defaults make an unset language indistinguishable from saved English.
    // Older backends ignore the option and keep returning English as before.
    return getHermesConfigRecord(profile, { includeDefaults: false })
  },
  saveConfig: (config, profile) => {
    if (typeof window === 'undefined' || !window.hermesDesktop?.api) {
      return Promise.resolve({ ok: true })
    }

    return saveHermesConfig(config, profile, { preserveLanguage: true })
  }
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
  /**
   * Config scope the persisted locale is read from and written to.
   *
   * The window's active profile settles AFTER mount — `$activeGatewayProfile`
   * boots as `'default'` while the backend pool resolves the real one — and a
   * profile-less request lands on whichever profile is ambient at that moment.
   * Reading the locale against that placeholder, then persisting an explicit
   * pick against the settled profile, is how the user's choice gets written to
   * a config nothing ever reads back (#113980). Passing the profile in keeps
   * both ends on one scope, and re-reads when it changes.
   */
  profile?: string | null
}

export function I18nProvider({
  children,
  configClient = defaultConfigClient,
  initialLocale,
  profile
}: I18nProviderProps) {
  const [locale, setLocaleState] = useState<Locale>(() => normalizeLocale(initialLocale))
  const [isLoadingConfig, setIsLoadingConfig] = useState(false)
  const [isSavingLocale, setIsSavingLocale] = useState(false)
  const [configLoadError, setConfigLoadError] = useState<Error | null>(null)
  const [saveError, setSaveError] = useState<Error | null>(null)
  const localeRef = useRef(locale)
  // Set once the user picks a language through setLocale: a startup read that
  // resolves (or fails) after that must never overwrite an explicit choice.
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
    let retryTimer: ReturnType<typeof setTimeout> | null = null
    let retryCount = 0

    // The desktop races its own backend at startup: the renderer mounts before
    // the backend is ready, so the first /api/config call can time out. We keep
    // the established permanent-failure contract — a rejected config load
    // settles on English so the UI stays usable — but bounded retries recover
    // transient startup failures, applying the persisted display.language once
    // the backend comes up.
    //
    // `profile` is a dependency, not just a parameter: the window's profile
    // scope resolves after mount, and a read issued against the placeholder
    // scope can never agree with the write an explicit pick performs later
    // (#113980). Re-running here is what lets the settled scope win.
    const MAX_LOCALE_RETRIES = 10
    const LOCALE_RETRY_DELAY_MS = 3_000

    const loadLocale = () => {
      setIsLoadingConfig(true)
      setConfigLoadError(null)

      return configClient
        .getConfig(profile)
        .then(async config => {
          if (cancelled || userLocaleRef.current) {
            return
          }

          const saved = getConfigDisplayLanguage(config)

          // A saved choice needs no machine probe and always takes precedence.
          if (isSupportedLocaleValue(saved)) {
            setLocaleState(normalizeLocale(saved))

            return
          }

          // Keep inference unsaved so OS language changes apply on the next boot
          // until the user explicitly picks a language.
          const machineProfile = await window.hermesDesktop?.getMachineProfile?.().catch(() => null)

          if (!cancelled && !userLocaleRef.current) {
            setLocaleState(resolveInitialLocale(undefined, machineProfile?.locale))
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

      if (retryTimer) {
        clearTimeout(retryTimer)
      }
    }
  }, [configClient, initialLocale, profile])

  const setLocale = useCallback(
    async (next: Locale) => {
      const previousLocale = localeRef.current

      userLocaleRef.current = true
      setSaveError(null)
      setLocaleState(next)

      if (!configClient) {
        return
      }

      setIsSavingLocale(true)

      try {
        const latestConfig = await configClient.getConfig(profile)
        const result = await configClient.saveConfig(withConfigDisplayLanguage(latestConfig, next), profile)

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
    [configClient, profile]
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
