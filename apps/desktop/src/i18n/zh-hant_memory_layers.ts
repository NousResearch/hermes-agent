import type { MemoryLayersTranslations } from './types_memory_layers'

export const zhHantMemoryLayers: MemoryLayersTranslations = {
  activeProvider: '目前記憶提供者',
  noProvider: '無 — 僅使用內建記憶',
  noProviderDesc: 'Hermes 僅使用下方的內建層。在「記憶提供者」中選擇外部提供者即可新增。',
  providerDesc: name => `${name} 會在內建層之外，為每個工作階段補充自己的記憶召回。`,
  status: {
    ready: '就緒',
    needs_config: '需要設定',
    unavailable: '無法使用',
    missing: '未安裝'
  },
  builtinLayer: '內建層',
  alwaysOn: '永遠啟用',
  builtinDesc:
    '無論是否啟用外部提供者，MEMORY.md 和 USER.md 都會載入每個工作階段。提供者是對這一層的補充，絕不會取代它。'
}
