import type { MemoryDiscoveryTranslations } from './types_memory_discovery'

export const zhHantMemoryDiscovery = {
  installed: '已安裝',
  availableToInstall: '可安裝',
  installationRequired: '需要安裝',
  reviewInstall: '審核並安裝',
  exploreAll: '瀏覽全部…',
  missing: '缺失',
  installConsent: '安裝並啟用外掛及其依賴。目前的記憶供應商保持不變，直到你明確選擇使用它。',
  builtin: '內建',
  configureElsewhere: '請使用 CLI 設定，或更新 Hermes 以儲存設定而不啟用。',
  notReady: '請完成設定並安裝缺少的依賴。剛安裝後請重新啟動後端，再重試。',
  useFailed: '無法使用此供應商。請檢查設定後重試。',

  activeProvider: name => `使用中：${name}`,
  useProvider: '使用提供者',
  loadFailed: '無法載入記憶提供者',
  ownerChanged: '請切換回開啟此安裝程式時的連線和設定檔，然後重試。',
  notDiscovered: '套件已安裝，但尚未發現其記憶提供者。請返回記憶設定重新探索。',
  installedNotice: '已發現提供者。請先設定，再明確選擇使用。',
  backToMemory: '返回記憶設定',
  connect: '連線',
  reconnect: '重新連線',
  connectOAuth: '透過 OAuth 連線',
  apiKeySet: '已設定 API 金鑰',
  oauthSet: 'OAuth 已連線',
  waitingConsent: '正在等待瀏覽器授權…',
  stopWaiting: '停止等待',
  stoppedWaiting: '已停止等待。授權可能仍在進行中。',
  startFailed: '無法開始連線。',
  connectionFailed: '連線失敗。'
} satisfies Partial<MemoryDiscoveryTranslations>
