import type { TranslationOverrides } from './define-locale'

// Shell notices (remote-display toast, butterbar), spread into ja.ts.
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
      'このサーバーはこのメッセージの前回の送信を確認できなかったため、すでに実行された可能性があります。再送信する前に会話を確認してください。'
  }
} satisfies Pick<TranslationOverrides, 'remoteDisplayBanner' | 'butterbar' | 'promptNotices'>
