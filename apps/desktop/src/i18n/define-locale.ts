import { en } from './en'
import type { Translations } from './types'

type TranslationOverride<T> = T extends (...args: never[]) => string
  ? T
  : T extends readonly unknown[]
    ? T
    : T extends string
      ? string
      : T extends object
        ? { [K in keyof T]?: TranslationOverride<T[K]> }
        : T

// A few locale bundles still carry copy for UI surfaces that were removed from
// the current English contract. Keep those known legacy keys typed while the
// locale files catch up, without opening TranslationOverrides to arbitrary keys.
type LegacyTranslationOverrides = {
  titlebar?: {
    openKeybinds?: string
  }
  settings?: {
    gateway?: {
      appliesTo?: string
    }
  }
  preview?: {
    closeTab?: (label: string) => string
  }
}

export type TranslationOverrides = TranslationOverride<Translations> & LegacyTranslationOverrides

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function mergeTranslations<T>(base: T, overrides: TranslationOverride<T> | undefined): T {
  if (!isRecord(base) || !isRecord(overrides)) {
    return (overrides ?? base) as T
  }

  const result: Record<string, unknown> = { ...base }

  for (const [key, value] of Object.entries(overrides)) {
    if (value === undefined) {
      continue
    }

    const baseValue = result[key]
    result[key] = isRecord(baseValue) && isRecord(value) ? mergeTranslations(baseValue, value) : value
  }

  return result as T
}

export function defineLocale(overrides: TranslationOverrides): Translations {
  return mergeTranslations<Translations>(en, overrides)
}
