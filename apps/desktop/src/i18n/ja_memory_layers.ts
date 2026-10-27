import type { MemoryLayersTranslations } from './types_memory_layers'

export const jaMemoryLayers: MemoryLayersTranslations = {
  activeProvider: '有効なメモリプロバイダー',
  noProvider: 'なし — 組み込みメモリのみ',
  noProviderDesc:
    'Hermes は下の組み込みレイヤーのみを使用します。追加するには「設定 › メモリとコンテキスト」で外部プロバイダーを選択してください。',
  providerDesc: name => `${name} は組み込みレイヤーに加えて、各セッションに独自の想起を追加します。`,
  status: {
    ready: '準備完了',
    needs_config: '設定が必要',
    unavailable: '利用不可',
    missing: '未インストール'
  },
  builtinLayer: '組み込みレイヤー',
  alwaysOn: '常に有効',
  builtinDesc:
    'MEMORY.md と USER.md は、外部プロバイダーの有無にかかわらず毎セッション読み込まれます。プロバイダーはこのレイヤーに追加されるもので、置き換えることはありません。'
}
