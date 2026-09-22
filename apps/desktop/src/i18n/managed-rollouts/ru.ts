import { mergeTranslations, type TranslationOverride } from '@hermes/shared/i18n'

import { managedRolloutsEn } from './en'
import type { ManagedRolloutMessages } from '../managed-rollouts-types'

const overrides: TranslationOverride<ManagedRolloutMessages> = {
  title: 'Управляемые развертывания',
  noActive: 'Нет активного снимка развертывания.',
  history: 'История развертываний',
  unresolved: count => `Неразрешенные ограждения: ${count}`,
  archived: 'архивировано',
  active: 'активно'
}

export const managedRolloutsRu = mergeTranslations<ManagedRolloutMessages>(managedRolloutsEn, overrides)
export const managedRollouts = managedRolloutsRu
export default managedRolloutsRu
