import type { CatalogTranslations, ManagedRolloutMessages } from '../managed-rollouts-types'
import type { Locale } from '../types'

import { managedRolloutsAr } from './ar'
import { managedRolloutsEn } from './en'
import { managedRolloutsJa } from './ja'
import { managedRolloutsRu } from './ru'
import { managedRolloutsZh } from './zh'
import { managedRolloutsZhHant } from './zh-hant'

export const MANAGED_ROLLOUT_TRANSLATIONS: Record<Locale, ManagedRolloutMessages> = {
  en: managedRolloutsEn,
  zh: managedRolloutsZh,
  'zh-hant': managedRolloutsZhHant,
  ja: managedRolloutsJa,
  ar: managedRolloutsAr,
  ru: managedRolloutsRu
}

/** Read the namespace loaded by the catalog for the selected locale. */
export function getManagedRolloutMessages(translations: CatalogTranslations): ManagedRolloutMessages {
  return translations.settings.managedRollouts
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
