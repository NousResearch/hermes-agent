import type { MemoryLayersTranslations } from './types_memory_layers'

export const zhMemoryLayers: MemoryLayersTranslations = {
  activeProvider: '当前记忆提供方',
  noProvider: '无 — 仅使用内置记忆',
  noProviderDesc: 'Hermes 仅使用下方的内置层。在“设置 › 记忆与上下文”中选择外部提供方即可添加。',
  providerDesc: name => `${name} 会在内置层之外，为每个会话补充自己的记忆召回。`,
  status: {
    ready: '就绪',
    needs_config: '需要设置',
    unavailable: '不可用',
    missing: '未安装'
  },
  builtinLayer: '内置层',
  alwaysOn: '始终启用',
  builtinDesc: '无论是否启用外部提供方，MEMORY.md 和 USER.md 都会载入每个会话。提供方是对这一层的补充，绝不会取代它。'
}
