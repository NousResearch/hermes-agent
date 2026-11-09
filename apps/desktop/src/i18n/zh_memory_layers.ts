import type { MemoryLayersTranslations } from './types_memory_layers'

export const zhMemoryLayers: MemoryLayersTranslations = {
  activeProvider: '当前记忆提供方',
  selectedProvider: '已选记忆提供方',
  selectedDesc: current => `尚未生效：在保存此选择之前，会话仍使用 ${current}。`,
  builtinMemory: '仅内置记忆',
  noProvider: '无 — 仅内置记忆',
  noProviderDesc: 'Hermes 仅使用下方的内置层。如需添加，请在 设置 › 记忆与上下文 中选择外部提供方。',
  providerDesc: name => `${name} 会在内置层之外，为每个会话添加自己的回忆。`,
  providerNotReadyDesc: name => `${name} 是已保存的提供方，但在就绪之前不会添加回忆。内置层在任何情况下都可用。`,
  providerUnknownDesc: name => `${name} 是已保存的提供方；Hermes 无法读取它是否就绪。`,
  statusUnavailable: '记忆状态不可用',
  statusUnavailableDesc: '无法读取当前记忆提供方及其是否就绪。',
  retry: '重试',
  status: {
    ready: '就绪',
    needs_config: '需要设置',
    unavailable: '不可用',
    missing: '未安装'
  },
  builtinLayer: '内置层',
  onByDefault: '默认开启',
  on: '开启',
  off: '关闭',
  onlyFile: file => `仅 ${file}`,
  builtinDesc:
    'MEMORY.md 和 USER.md 是 Hermes 自己的记忆，独立于任何外部提供方。两者默认都会载入会话；可在 设置 › 记忆与上下文 中用记忆和用户档案开关分别关闭。提供方是对这一层的补充，绝不会取代它。'
}
