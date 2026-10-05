import { isRecord } from '@hermes/shared/i18n'
import { applyDocumentLocale } from '@hermes/shared/document-locale'
import { useStore } from '@nanostores/react'
import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'

import { getHermesConfigRecord, type HermesConfigRecord, retainConfigReadOrigin, saveHermesConfig } from '@/hermes'

import { TRANSLATIONS } from './catalog'
import {
  DEFAULT_LOCALE,
  isRtlLocale,
  isSupportedLocaleValue,
  localeConfigValue,
  normalizeLocale,
  resolveInitialLocale
} from './languages'
import { $appLocaleVersion, normalizeLocaleId, resolveTranslations } from './registry'
import { $requestedLocale, setRuntimeI18nLocale } from './runtime'
import type { Locale, Translations } from './types'

export interface I18nConfigClient {
  getConfig: () => Promise<HermesConfigRecord>
  saveConfig: (config: HermesConfigRecord) => Promise<{ ok: boolean }>
}

const defaultConfigClient: I18nConfigClient = {
  getConfig: () => {
    if (typeof window === 'undefined' || !window.hermesDesktop?.api) {
      return Promise.resolve({})
    }

    // Merged defaults make an unset language indistinguishable from saved English.
    // Older backends ignore the option and keep returning English as before.
    return getHermesConfigRecord(undefined, { includeDefaults: false })
  },
  saveConfig: config => {
    if (typeof window === 'undefined' || !window.hermesDesktop?.api) {
      return Promise.resolve({ ok: true })
    }

    // No explicit scope: saveHermesConfig resolves the record's captured read
    // origin itself (resolveConfigWriteScope), and withConfigDisplayLanguage
    // retains that origin onto the derived record.
    return saveHermesConfig(config, undefined, { preserveLanguage: true })
  }
}

export function getConfigDisplayLanguage(config: HermesConfigRecord): unknown {
  return isRecord(config.display) ? config.display.language : undefined
}

export function withConfigDisplayLanguage(config: HermesConfigRecord, locale: Locale): HermesConfigRecord {
  const display = isRecord(config.display) ? config.display : {}

  return retainConfigReadOrigin(
    {
      ...config,
      display: {
        ...display,
        language: localeConfigValue(locale)
      }
    },
    config
  )
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
  /** Reconcile when the window's config owner changes, without remounting its children. */
  scopeKey?: string
}

export function I18nProvider({
  children,
  configClient = defaultConfigClient,
  initialLocale,
  scopeKey
}: I18nProviderProps) {
  const [locale, setLocaleState] = useState<Locale>(() => normalizeLocale(initialLocale))
  const [isLoadingConfig, setIsLoadingConfig] = useState(false)
  const [isSavingLocale, setIsSavingLocale] = useState(false)
  const [configLoadError, setConfigLoadError] = useState<Error | null>(null)
  const [saveError, setSaveError] = useState<Error | null>(null)
  const localeRef = useRef(locale)
  // An explicit pick beats a late read in its own scope, not in other profiles.
  const userLocaleRef = useRef(false)
  const scopeGenerationRef = useRef(0)
  // Registered languages (plugin packs, backend `.desktop.yaml`) change the
  // catalog without a locale change; the version keys re-resolution.
  const registryVersion = useStore($appLocaleVersion)
  // The language THIS scope's config asked for (what $requestedLocale was last
  // set to from here). A ref, not the atom: another provider's or an earlier
  // scope's ask must never be promoted into this one.
  const requestedRef = useRef<null | string>(null)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    localeRef.current = locale
    setRuntimeI18nLocale(locale)
    applyDocumentLocale(locale, isRtlLocale(locale))
  }, [locale, registryVersion])

  // A saved `display.language` the app could not render at read time (`pl`
  // before its pack arrived) is parked in $requestedLocale; the moment a
  // registration makes it renderable, promote it — unless the user has since
  // picked something else in this scope.
  useEffect(() => {
    const requested = requestedRef.current

    if (!requested || userLocaleRef.current || !isSupportedLocaleValue(requested)) {
      return
    }

    const next = normalizeLocale(requested)

    if (next !== localeRef.current) {
      setLocaleState(next)
    }
  }, [registryVersion])

  // eslint-disable-next-line no-restricted-syntax -- scope-local request generation and user intent, not an atom mirror
  useEffect(() => {
    scopeGenerationRef.current += 1
    userLocaleRef.current = false
    requestedRef.current = null
    setSaveError(null)
    setIsSavingLocale(false)

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
    const MAX_LOCALE_RETRIES = 10
    const LOCALE_RETRY_DELAY_MS = 3_000

    const loadLocale = () => {
      setIsLoadingConfig(true)
      setConfigLoadError(null)

      return configClient
        .getConfig()
        .then(async config => {
          if (cancelled || userLocaleRef.current) {
            return
          }

          const saved = getConfigDisplayLanguage(config)

          // Publish the raw ask even when it names a language only a pack
          // can render; the backend-pack sync fetches that pack by this id.
          requestedRef.current = typeof saved === 'string' && saved.trim() ? normalizeLocaleId(saved) : null
          $requestedLocale.set(requestedRef.current)

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
      scopeGenerationRef.current += 1

      if (retryTimer) {
        clearTimeout(retryTimer)
      }
    }
  }, [configClient, initialLocale, scopeKey])

  const setLocale = useCallback(
    async (next: Locale) => {
      const previousLocale = localeRef.current
      const generation = scopeGenerationRef.current

      userLocaleRef.current = true
      setSaveError(null)
      setLocaleState(next)
      requestedRef.current = next
      $requestedLocale.set(next)

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

        // The write still belongs to the GET's captured origin, but its late
        // outcome must not roll another profile's chrome back to this one.
        if (generation === scopeGenerationRef.current) {
          setLocaleState(previousLocale)
          setSaveError(nextError)
        }

        throw nextError
      } finally {
        if (generation === scopeGenerationRef.current) {
          setIsSavingLocale(false)
        }
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
      t: resolveTranslations(locale)
    }),
    // `registryVersion` is the registry's change token: a pack landing after
    // first paint re-resolves `t` without a locale change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [configLoadError, isLoadingConfig, isSavingLocale, locale, registryVersion, saveError, setLocale]
  )

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n(): I18nContextValue {
  return useContext(I18nContext)
}
