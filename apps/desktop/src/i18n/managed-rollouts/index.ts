import type { Locale, Translations } from '../types'
import { managedRolloutsAr } from './ar'
import { managedRolloutsEn } from './en'
import { managedRolloutsJa } from './ja'
import { managedRolloutsRu } from './ru'
import { managedRolloutsZh } from './zh'
import { managedRolloutsZhHant } from './zh-hant'
import type { ManagedRolloutMessages } from '../managed-rollouts-types'

export const MANAGED_ROLLOUT_TRANSLATIONS: Record<Locale, ManagedRolloutMessages> = {
  en: managedRolloutsEn,
  zh: managedRolloutsZh,
  'zh-hant': managedRolloutsZhHant,
  ja: managedRolloutsJa,
  ar: managedRolloutsAr,
  ru: managedRolloutsRu
}

/**
 * Resolve the expanded namespace beside the legacy six-key catalog seam.
 * The legacy locale godfiles remain source-compatible; managed-rollout UI gets
 * the expanded copy through the same locale selected by useI18n.
 */
export function getManagedRolloutMessages(
  _translations: Translations,
  locale: Locale = 'en'
): ManagedRolloutMessages {
  return MANAGED_ROLLOUT_TRANSLATIONS[locale] ?? MANAGED_ROLLOUT_TRANSLATIONS.en
}

export {
  managedRolloutsAr,
  managedRolloutsEn,
  managedRolloutsJa,
  managedRolloutsRu,
  managedRolloutsZh,
  managedRolloutsZhHant
}
export type { ManagedRolloutMessages } from '../managed-rollouts-types'
