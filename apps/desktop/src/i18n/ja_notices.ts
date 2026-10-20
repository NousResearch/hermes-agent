import type { TranslationOverrides } from './define-locale'
import { jaSessionCommands } from './ja_session_commands'

// Shell notices (remote-display toast, butterbar, keep-awake offer), spread into ja.ts.
export const jaNotices = {
  remoteDisplayBanner: {
    message: reason =>
      `ソフトウェアレンダリングが有効です — リモートディスプレイを検出しました（${reason}）。ちらつきを防ぐため GPU アクセラレーションは無効化されています。`
  },
  butterbar: {
    goTo: (index, total) => `お知らせ ${index} / ${total} を表示`,
    legal: {
      before: 'Hermes Agent のご利用には',
      terms: '利用規約',
      between: 'および',
      privacy: 'プライバシーポリシー',
      after: 'が適用されます。'
    }
  },
  promptNotices: {
    legacySendUnconfirmed:
      'このサーバーはこのメッセージの前回の送信を確認できなかったため、すでに実行された可能性があります。再送信する前に会話を確認してください。',
    mcpReloadDeferred:
      'このチャットは処理中です。MCP の変更は現在のターンが終わって MCP を再読み込みしたとき、またはチャットを開き直したときに反映されます',
    mcpReloadNotInChat: server => `${server} はこのチャットでは有効になっていません`,
    mcpReloadUnavailable: servers => `接続できませんでした: ${servers}`,
    compactedTurnUnavailable: 'このメッセージはチャットの圧縮時に要約されたため、復元や編集はできなくなりました。',
    staleTurnUnavailable: 'このメッセージはこのチャットの履歴にはもうありません。',
    sessionCommands: jaSessionCommands
  },
  keepAwakeOffer: {
    title: 'Hermes の作業中にコンピューターがスリープしました',
    message:
      'タスクの実行中は、Hermes がコンピューターのスリープを防げます。画面はいつもどおりオフになり、バッテリー駆動中はスリープすることがあります。',
    action: '実行中はスリープしない'
  },
  billingBlock: {
    titleNous: 'Nous クレジットが不足しています',
    titleProvider: provider => `クレジット不足 — ${provider}`,
    fallbackMessage: 'アカウントのクレジットが不足しています。続行するにはクレジットを追加してください。',
    openBilling: '請求を開く',
    addCredits: 'クレジットを追加',
    dismiss: '閉じる'
  }
} satisfies Pick<
  TranslationOverrides,
  'remoteDisplayBanner' | 'butterbar' | 'promptNotices' | 'keepAwakeOffer' | 'billingBlock'
>
