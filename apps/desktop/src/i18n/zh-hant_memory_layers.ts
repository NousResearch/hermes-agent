import type { MemoryLayersTranslations } from './types_memory_layers'

export const zhHantMemoryLayers: MemoryLayersTranslations = {
  activeProvider: '目前的記憶提供者',
  selectedProvider: '已選的記憶提供者',
  selectedDesc: current => `尚未生效：在儲存此選擇之前，工作階段仍使用 ${current}。`,
  builtinMemory: '僅內建記憶',
  noProvider: '無 — 僅內建記憶',
  noProviderDesc: 'Hermes 只使用下方的內建層。如需新增，請在 設定 › 記憶與脈絡 中選擇外部提供者。',
  providerDesc: name => `${name} 會在內建層之外，為每個工作階段加入自己的回憶。`,
  providerNotReadyDesc: name => `${name} 是已儲存的提供者，但在就緒之前不會加入回憶。內建層在任何情況下都可運作。`,
  providerUnknownDesc: name => `${name} 是已儲存的提供者；Hermes 無法讀取它是否就緒。`,
  statusUnavailable: '記憶狀態無法取得',
  statusUnavailableDesc: '無法讀取目前的記憶提供者及其是否就緒。',
  retry: '重試',
  status: {
    ready: '就緒',
    needs_config: '需要設定',
    unavailable: '無法使用',
    missing: '未安裝'
  },
  builtinLayer: '內建層',
  onByDefault: '預設開啟',
  on: '開啟',
  off: '關閉',
  onlyFile: file => `僅 ${file}`,
  builtinDesc:
    'MEMORY.md 和 USER.md 是 Hermes 自己的記憶，獨立於任何外部提供者。兩者預設都會載入工作階段；可在 設定 › 記憶與脈絡 中以記憶和使用者檔案開關分別關閉。提供者是對這一層的補充，絕不會取代它。'
}
