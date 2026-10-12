import type { TranslationOverride } from '@hermes/shared/i18n'

import type { SystemResourcesTranslations } from './types_system_resources'

export const jaSystemResources: TranslationOverride<SystemResourcesTranslations> = {
  title: 'システムリソース',
  loading: 'リソース…',
  gpuUtilization: 'GPU 使用率',
  gpuMemory: 'GPU メモリ',
  ram: 'RAM',
  cpu: 'CPU',
  disk: 'ディスク',
  unifiedNote: 'ユニファイドメモリ——GPU とシステムがこのプールを共有します。',
  toggle: 'システムリソース'
}
