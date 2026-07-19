import registry from '../../../locales/registry.json' with { type: 'json' }

export type Locale = keyof typeof registry.locales

export interface LocaleMetadata {
  name: string
  triggerLabel: string
}

const hasOwn = (value: object, key: string) => Object.prototype.hasOwnProperty.call(value, key)

export const LOCALES = Object.freeze(Object.keys(registry.locales) as Locale[])
export const DEFAULT_LOCALE = registry.default as Locale
export const LOCALE_METADATA = registry.locales as Record<Locale, LocaleMetadata>

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

  // Chinese product languages are explicit. Unknown zh-* inputs must not be
  // guessed into Simplified or Traditional Chinese.
  if (normalized.startsWith('zh-')) {
    return null
  }

  const primary = normalized.split('-', 1)[0]

  return isLocale(primary) ? primary : null
}
