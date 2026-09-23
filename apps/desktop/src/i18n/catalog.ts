import { mergeTranslations } from '@hermes/shared/i18n'

import { ar } from './ar'
import { en } from './en'
import { ja } from './ja'
import { MANAGED_ROLLOUT_TRANSLATIONS } from './managed-rollouts'
import type { CatalogTranslations } from './managed-rollouts-types'
import { ru } from './ru'
import type { Locale, Translations } from './types'
import { zh } from './zh'
import { zhHant } from './zh-hant'

function withManagedRollouts(base: Translations, locale: Locale): CatalogTranslations {
  return mergeTranslations(base as CatalogTranslations, {
    settings: { managedRollouts: MANAGED_ROLLOUT_TRANSLATIONS[locale] }
  })
}

export const TRANSLATIONS: Record<Locale, CatalogTranslations> = {
  en: withManagedRollouts(en, 'en'),
  zh: withManagedRollouts(zh, 'zh'),
  'zh-hant': withManagedRollouts(zhHant, 'zh-hant'),
  ja: withManagedRollouts(ja, 'ja'),
  ar: withManagedRollouts(ar, 'ar'),
  ru: withManagedRollouts(ru, 'ru')
}
