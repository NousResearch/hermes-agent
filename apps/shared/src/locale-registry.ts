import registry from '../../../locales/registry.json' with { type: 'json' }

export type Locale = keyof typeof registry.locales

export interface LocaleMetadata {
  name: string
  triggerLabel: string
  intlTag?: string
}

const hasOwn = (value: object, key: string) => Object.prototype.hasOwnProperty.call(value, key)

export const LOCALES = Object.freeze(Object.keys(registry.locales) as Locale[])
export const DEFAULT_LOCALE = registry.default as Locale
export const LOCALE_METADATA = registry.locales as Record<Locale, LocaleMetadata>

/** BCP-47 tag for Intl APIs; language-specific compatibility stays in the registry. */
export function localeIntlTag(locale: Locale): string {
  return LOCALE_METADATA[locale].intlTag ?? locale
}

const isLocale = (value: string): value is Locale => hasOwn(registry.locales, value)

const assertRegistryTargets = (entries: Record<string, string>, source: string) => {
  for (const [input, target] of Object.entries(entries)) {
    if (!isLocale(target)) {
      throw new Error(`${source} maps ${input} to unknown locale ${target}`)
    }
  }
}

if (!isLocale(DEFAULT_LOCALE)) {
  throw new Error(`locale registry default is not registered: ${registry.default}`)
}

assertRegistryTargets(registry.aliases, 'locale aliases')
assertRegistryTargets(registry.compatibilityAliases, 'locale compatibility aliases')

/** Normalize configuration, browser, and protocol input at the shared boundary. */
export function normalizeLocaleInput(value: unknown): Locale | null {
  if (typeof value !== 'string') {
    return null
  }

  const normalized = value.trim().toLowerCase().replace(/_/g, '-').replace(/\s+/g, '-')

  if (!normalized) {
    return null
  }

  if (isLocale(normalized)) {
    return normalized
  }

  const alias = (registry.aliases as Record<string, Locale>)[normalized]

  if (alias) {
    return alias
  }

  const compatibilityAlias = (registry.compatibilityAliases as Record<string, Locale>)[normalized]

  if (compatibilityAlias) {
    return compatibilityAlias
  }

  const primary = normalized.split('-', 1)[0]

  if (!isLocale(primary)) {
    return null
  }

  // A primary-subtag fallback is safe only when the registry has one product
  // language in that family. If multiple independent packs share the primary
  // tag, callers must use a registered value or an explicit compatibility
  // alias instead of guessing which pack owns the input.
  const hasSiblingPack = LOCALES.some(
    locale => locale !== primary && locale.startsWith(`${primary}-`)
  )

  return hasSiblingPack ? null : primary
}
