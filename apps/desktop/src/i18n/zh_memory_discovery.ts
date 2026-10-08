import type { MemoryDiscoveryTranslations } from './types_memory_discovery'

export const zhMemoryDiscovery = {
  installed: '已安装',
  availableToInstall: '可安装',
  installationRequired: '需要安装',
  reviewInstall: '审核并安装',
  exploreAll: '浏览全部…',
  missing: '缺失',
  installConsent: '安装并启用插件及其依赖。当前记忆提供商保持不变，直到你明确选择使用它。',
  builtin: '内置',
  configureElsewhere: '请使用 CLI 配置，或更新 Hermes 以保存设置而不启用。',
  notReady: '请完成配置并安装缺失依赖。刚安装后请重启后端，再重试。',
  useFailed: '无法使用此提供商。请检查配置后重试。',

  activeProvider: name => `使用中：${name}`,
  useProvider: '使用提供商',
  loadFailed: '无法加载记忆提供商',
  ownerChanged: '请切换回打开此安装程序时的连接和配置档案，然后重试。',
  notDiscovered: '软件包已安装，但尚未发现其记忆提供商。请返回记忆设置以重新发现。',
  installedNotice: '已发现提供商。请先配置，再明确选择使用。',
  backToMemory: '返回记忆设置',
  connect: '连接',
  reconnect: '重新连接',
  connectOAuth: '通过 OAuth 连接',
  apiKeySet: '已设置 API 密钥',
  oauthSet: 'OAuth 已连接',
  waitingConsent: '正在等待浏览器授权…',
  stopWaiting: '停止等待',
  stoppedWaiting: '已停止等待。授权可能仍在进行中。',
  startFailed: '无法开始连接。',
  connectionFailed: '连接失败。'
} satisfies Partial<MemoryDiscoveryTranslations>
