import { mergeTranslations, type TranslationOverride } from '@hermes/shared/i18n'

import { en, type Translations } from './en'

export type TranslationOverrides = TranslationOverride<Translations>

export const defineLocale = (overrides: TranslationOverrides): Translations =>
  mergeTranslations<Translations>(en, overrides)
