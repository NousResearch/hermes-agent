import { catalogs } from './catalogs'
import { translate, type TranslationValues } from './format'
import { DEFAULT_DESKTOP_LANGUAGE, type DesktopLanguage } from './languages'

let runtimeLanguage: DesktopLanguage = DEFAULT_DESKTOP_LANGUAGE

export function setRuntimeI18nLanguage(language: DesktopLanguage) {
  runtimeLanguage = language
}

export function translateNow(key: string, values?: TranslationValues): string {
  return translate(catalogs, runtimeLanguage, key, values)
}
