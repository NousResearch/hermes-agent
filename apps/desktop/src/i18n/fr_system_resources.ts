import type { TranslationOverride } from '@hermes/shared/i18n'

import type { SystemResourcesTranslations } from './types_system_resources'

export const frSystemResources: TranslationOverride<SystemResourcesTranslations> = {
  title: 'Ressources système',
  loading: 'Ressources…',
  gpuUtilization: 'Utilisation du GPU',
  gpuMemory: 'Mémoire GPU',
  ram: 'RAM',
  cpu: 'CPU',
  disk: 'Disque',
  unifiedNote: 'Mémoire unifiée — le GPU et le système partagent cet espace.',
  toggle: 'Ressources système'
}
