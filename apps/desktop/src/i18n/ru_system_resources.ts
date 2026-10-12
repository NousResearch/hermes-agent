import type { TranslationOverride } from '@hermes/shared/i18n'

import type { SystemResourcesTranslations } from './types_system_resources'

export const ruSystemResources: TranslationOverride<SystemResourcesTranslations> = {
  title: 'Системные ресурсы',
  loading: 'Ресурсы…',
  gpuUtilization: 'Загрузка GPU',
  gpuMemory: 'Память GPU',
  ram: 'ОЗУ',
  cpu: 'ЦП',
  disk: 'Диск',
  unifiedNote: 'Единая память — GPU и система используют общий пул.',
  toggle: 'Системные ресурсы'
}
