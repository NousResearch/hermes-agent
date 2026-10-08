import type { MemoryDiscoveryTranslations } from './types_memory_discovery'

export const jaMemoryDiscovery = {
  installed: 'インストール済み',
  availableToInstall: 'インストール可能',
  installationRequired: 'インストールが必要です',
  reviewInstall: '確認してインストール',
  exploreAll: 'すべて見る…',
  missing: '見つかりません',
  installConsent:
    '依存関係とともにプラグインをインストールして有効にします。メモリープロバイダーは明示的に選択するまで変更されません。',
  builtin: '組み込み',
  configureElsewhere: 'CLIで設定するか、選択を変更せず保存できるHermesに更新してください。',
  notReady: '設定と依存関係を確認してください。インストール直後はバックエンドを再起動して再試行してください。',
  useFailed: '使用できませんでした。設定を確認して再試行してください。',

  activeProvider: name => `使用中: ${name}`,
  useProvider: 'このプロバイダーを使う',
  loadFailed: 'メモリプロバイダーを読み込めませんでした',
  ownerChanged: 'この画面を開いた接続とプロファイルに戻ってから、もう一度お試しください。',
  notDiscovered:
    'パッケージはインストールされましたが、メモリプロバイダーはまだ検出されていません。メモリ設定に戻って再検出してください。',
  installedNotice: 'プロバイダーが見つかりました。設定後、明示的に使用を選択してください。',
  backToMemory: 'メモリ設定に戻る',
  connect: '接続',
  reconnect: '再接続',
  connectOAuth: 'OAuthで接続',
  apiKeySet: 'APIキー設定済み',
  oauthSet: 'OAuth接続済み',
  waitingConsent: 'ブラウザーでの同意を待っています…',
  stopWaiting: '待機をやめる',
  stoppedWaiting: '待機を停止しました。承認はまだ保留中の可能性があります。',
  startFailed: '接続を開始できませんでした。',
  connectionFailed: '接続に失敗しました。'
} satisfies Partial<MemoryDiscoveryTranslations>
