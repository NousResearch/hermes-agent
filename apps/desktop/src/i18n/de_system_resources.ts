import type { TranslationOverride } from '@hermes/shared/i18n'

import type { SystemResourcesTranslations } from './types_system_resources'

export const deSystemResources: TranslationOverride<SystemResourcesTranslations> = {
  title: 'Systemressourcen',
  loading: 'Ressourcen…',
  gpuUtilization: 'GPU-Auslastung',
  gpuMemory: 'GPU-Speicher',
  ram: 'RAM',
  cpu: 'CPU',
  disk: 'Datenträger',
  unifiedNote: 'Unified Memory — die GPU und das System teilen sich diesen Speicherpool.',
  toggle: 'Systemressourcen'
}
