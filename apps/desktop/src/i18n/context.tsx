import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'

import { getHermesConfigRecord, type HermesConfigRecord, saveHermesConfig } from '@/hermes'

import { catalogs as defaultCatalogs } from './catalogs'
import { createTranslator, type TranslationCatalogs, type TranslationValues } from './format'
import { DEFAULT_DESKTOP_LANGUAGE, type DesktopLanguage, desktopLanguageConfigValue, normalizeDesktopLanguage } from './languages'
import { setRuntimeI18nLanguage } from './runtime'

export type Translate = (key: string, values?: TranslationValues) => string

export interface I18nConfigClient {
  getConfig: () => Promise<HermesConfigRecord>
  saveConfig: (config: HermesConfigRecord) => Promise<{ ok: boolean }>
}

export interface I18nContextValue {
  configLoadError: Error | null
  isLoadingConfig: boolean
  isSavingLanguage: boolean
  language: DesktopLanguage
  saveError: Error | null
  setLanguage: (language: DesktopLanguage) => Promise<void>
  t: Translate
}

const I18nContext = createContext<I18nContextValue | null>(null)

const defaultConfigClient: I18nConfigClient = {
  getConfig: getHermesConfigRecord,
  saveConfig: saveHermesConfig
}

export interface I18nProviderProps {
  catalogs?: TranslationCatalogs
  children: ReactNode
  configClient?: I18nConfigClient | null
  initialLanguage?: unknown
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function getConfigDisplayLanguage(config: HermesConfigRecord): unknown {
  return isRecord(config.display) ? config.display.language : undefined
}

export function withConfigDisplayLanguage(
  config: HermesConfigRecord,
  language: DesktopLanguage
): HermesConfigRecord {
  const display = isRecord(config.display) ? config.display : {}

  return {
    ...config,
    display: {
      ...display,
      language: desktopLanguageConfigValue(language)
    }
  }
}

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error))
}

export function I18nProvider({
  catalogs = defaultCatalogs,
  children,
  configClient = defaultConfigClient,
  initialLanguage
}: I18nProviderProps) {
  const [language, setLanguageState] = useState<DesktopLanguage>(() => normalizeDesktopLanguage(initialLanguage))
  const [isLoadingConfig, setIsLoadingConfig] = useState(false)
  const [isSavingLanguage, setIsSavingLanguage] = useState(false)
  const [configLoadError, setConfigLoadError] = useState<Error | null>(null)
  const [saveError, setSaveError] = useState<Error | null>(null)
  const languageRef = useRef(language)

  useEffect(() => {
    languageRef.current = language
    setRuntimeI18nLanguage(language)
  }, [language])

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
          setLanguageState(normalizeDesktopLanguage(getConfigDisplayLanguage(config)))
        }
      })
      .catch(error => {
        if (!cancelled) {
          setConfigLoadError(toError(error))
          setLanguageState(DEFAULT_DESKTOP_LANGUAGE)
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
  }, [configClient, initialLanguage])

  const setLanguage = useCallback(
    async (nextLanguage: DesktopLanguage) => {
      const previousLanguage = languageRef.current

      setSaveError(null)
      setLanguageState(nextLanguage)

      if (!configClient) {
        return
      }

      setIsSavingLanguage(true)

      try {
        const latestConfig = await configClient.getConfig()
        const result = await configClient.saveConfig(withConfigDisplayLanguage(latestConfig, nextLanguage))

        if (!result.ok) {
          throw new Error('Failed to save language')
        }
      } catch (error) {
        const nextError = toError(error)

        setLanguageState(previousLanguage)
        setSaveError(nextError)

        throw nextError
      } finally {
        setIsSavingLanguage(false)
      }
    },
    [configClient]
  )

  const t = useMemo(() => createTranslator(catalogs, language), [catalogs, language])

  const value = useMemo<I18nContextValue>(
    () => ({
      configLoadError,
      isLoadingConfig,
      isSavingLanguage,
      language,
      saveError,
      setLanguage,
      t
    }),
    [configLoadError, isLoadingConfig, isSavingLanguage, language, saveError, setLanguage, t]
  )

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n(): I18nContextValue {
  const context = useContext(I18nContext)

  if (!context) {
    throw new Error('useI18n must be used within I18nProvider')
  }

  return context
}

export function useTranslation(): Translate {
  return useI18n().t
}
