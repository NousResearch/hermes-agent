import type { TranslationOverride } from '@hermes/shared/i18n'

import type { SystemResourcesTranslations } from './types_system_resources'

export const zhSystemResources: TranslationOverride<SystemResourcesTranslations> = {
  title: '系统资源',
  loading: '资源…',
  gpuUtilization: 'GPU 利用率',
  gpuMemory: '显存',
  ram: '内存',
  cpu: 'CPU',
  disk: '磁盘',
  unifiedNote: '统一内存——GPU 与系统共享此内存池。',
  toggle: '系统资源'
}
