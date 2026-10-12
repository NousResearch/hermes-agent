import type { TranslationOverride } from '@hermes/shared/i18n'

import type { SystemResourcesTranslations } from './types_system_resources'

export const esSystemResources: TranslationOverride<SystemResourcesTranslations> = {
  title: 'Recursos del sistema',
  loading: 'Recursos…',
  gpuUtilization: 'Uso de GPU',
  gpuMemory: 'Memoria de GPU',
  ram: 'RAM',
  cpu: 'CPU',
  disk: 'Disco',
  unifiedNote: 'Memoria unificada: la GPU y el sistema comparten este espacio.',
  toggle: 'Recursos del sistema'
}
