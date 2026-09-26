import { mergeTranslations, type TranslationOverride } from '@hermes/shared/i18n'

import { en } from './en'
import type { Translations } from './types'

// These keys existed in the Ukrainian bundle before their desktop surfaces were
// removed. Keep the migration allowance scoped to the known legacy copy rather
// than weakening TranslationOverride for every locale.
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

export const defineLocale = (overrides: TranslationOverrides): Translations =>
  mergeTranslations<Translations>(en, overrides)
