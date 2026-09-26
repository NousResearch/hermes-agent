import { localeDirection, normalizeLocaleInput } from './locale-registry.js'

/** Mirror the active locale onto `<html lang dir>`. No-op without a document (SSR, tests). */
export function applyDocumentLocale(locale: string): void {
  if (typeof document === 'undefined') {
    return
  }

  document.documentElement.lang = locale
  document.documentElement.dir = localeDirection(normalizeLocaleInput(locale) ?? 'en')
}
