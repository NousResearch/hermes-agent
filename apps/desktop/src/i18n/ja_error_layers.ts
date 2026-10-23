import type { Translations } from './types'

type ThreadCopy = Translations['assistant']['thread']

// The error card's layer fallbacks (lib/error-surface-copy.ts): used when a failure code has no own entry.
// Partial like the rest of this locale: an untranslated layer falls back to English.
export const jaErrorLayers: {
  [K in 'errorLayerBodies' | 'errorLayers']: Partial<ThreadCopy[K]>
} = {
  errorLayers: {
    auth: '認証エラー',
    billing: 'クレジット不足',
    disk: 'ディスク容量不足',
    endpoint: 'カスタムエンドポイントのエラー',
    gateway: 'ゲートウェイのエラー',
    generic: 'ターンが失敗しました',
    provider: 'プロバイダーのエラー',
    runtime: 'ローカルランタイムのエラー',
    streaming: 'ストリーミング接続のエラー'
  },
  errorLayerBodies: {
    generic:
      'Hermes の返信中に問題が発生しました。再試行してください。問題が続く場合はエラー詳細をコピーしてください。',
    provider:
      'AI サービスがリクエストを完了できませんでした。少し待って再試行するか、プロバイダーを切り替えてください。',
    endpoint: 'カスタムモデルサーバーに接続できません。サーバーが起動しているか確認し、メッセージを再送してください。',
    streaming: '返信が完了する前に接続が切れました。再試行してもう一度送信してください。'
  }
}
