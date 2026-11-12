import type { MemoryLayersTranslations } from './types_memory_layers'

export const jaMemoryLayers: MemoryLayersTranslations = {
  activeProvider: '有効なメモリプロバイダー',
  selectedProvider: '選択中のメモリプロバイダー',
  selectedDesc: current => `まだ有効ではありません。この選択が保存されるまで、セッションは ${current} を使い続けます。`,
  builtinMemory: '組み込みメモリのみ',
  noProvider: 'なし — 組み込みメモリのみ',
  noProviderDesc:
    'Hermes は下の組み込みレイヤーのみを使用します。追加するには、設定 › メモリとコンテキストで外部プロバイダーを選択してください。',
  providerDesc: name => `${name} は組み込みレイヤーに加えて、すべてのセッションに独自の想起を追加します。`,
  providerNotReadyDesc: name =>
    `${name} は保存済みのプロバイダーですが、準備ができるまで想起は追加されません。組み込みレイヤーはどちらの場合も機能します。`,
  providerUnknownDesc: name =>
    `${name} は保存済みのプロバイダーです。準備ができているかを Hermes は読み取れませんでした。`,
  statusUnavailable: 'メモリの状態を取得できません',
  statusUnavailableDesc: 'どのメモリプロバイダーが有効か、また準備ができているかを読み取れませんでした。',
  retry: '再試行',
  status: {
    ready: '準備完了',
    needs_config: '設定が必要',
    unavailable: '利用不可',
    missing: '未インストール'
  },
  builtinLayer: '組み込みレイヤー',
  onByDefault: '既定でオン',
  on: 'オン',
  off: 'オフ',
  onlyFile: file => `${file} のみ`,
  builtinDesc:
    'MEMORY.md と USER.md は外部プロバイダーとは別の Hermes 自身のメモリです。どちらも既定でセッションに読み込まれ、設定 › メモリとコンテキストのメモリとユーザープロファイルの切り替えでそれぞれオフにできます。プロバイダーはこのレイヤーに追加されるもので、置き換えることはありません。'
}
