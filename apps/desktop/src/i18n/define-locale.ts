import { mergeTranslations, type TranslationOverride } from '@hermes/shared/i18n'

import { en } from './en'
import type { Translations } from './types'

export type TranslationOverrides = TranslationOverride<Translations>

export const defineLocale = (overrides: TranslationOverrides): Translations => mergeTranslations<Translations>(en, overrides)
