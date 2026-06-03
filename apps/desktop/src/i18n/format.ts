import { DEFAULT_DESKTOP_LANGUAGE, type DesktopLanguage } from './languages'

export type TranslationValues = Record<string, number | string>
export type TranslationCatalog = Record<string, string>
export type TranslationCatalogs = Record<DesktopLanguage, TranslationCatalog>

export interface MissingTranslationDiagnostic {
  key: string
  language: DesktopLanguage
  reason: 'missing-active' | 'missing-default'
}

const missingTranslations = new Map<string, MissingTranslationDiagnostic>()

function recordMissingTranslation(diagnostic: MissingTranslationDiagnostic) {
  missingTranslations.set(`${diagnostic.language}:${diagnostic.key}:${diagnostic.reason}`, diagnostic)

  if (import.meta.env.DEV) {
    console.warn(`[i18n] Missing ${diagnostic.language} translation for "${diagnostic.key}"`)
  }
}

export function clearMissingTranslationDiagnostics() {
  missingTranslations.clear()
}

export function getMissingTranslationDiagnostics(): MissingTranslationDiagnostic[] {
  return [...missingTranslations.values()]
}

export function interpolate(template: string, values: TranslationValues = {}): string {
  return template.replace(/\{([A-Za-z0-9_.-]+)\}/g, (match, name: string) => {
    const value = values[name]

    return value == null ? match : String(value)
  })
}

export function translate(
  catalogs: TranslationCatalogs,
  language: DesktopLanguage,
  key: string,
  values?: TranslationValues
): string {
  const activeText = catalogs[language]?.[key]

  if (activeText != null) {
    return interpolate(activeText, values)
  }

  if (language !== DEFAULT_DESKTOP_LANGUAGE) {
    recordMissingTranslation({ key, language, reason: 'missing-active' })
  }

  const fallbackText = catalogs[DEFAULT_DESKTOP_LANGUAGE]?.[key]

  if (fallbackText != null) {
    return interpolate(fallbackText, values)
  }

  recordMissingTranslation({ key, language: DEFAULT_DESKTOP_LANGUAGE, reason: 'missing-default' })

  return key
}

export function createTranslator(catalogs: TranslationCatalogs, language: DesktopLanguage) {
  return (key: string, values?: TranslationValues): string => translate(catalogs, language, key, values)
}
