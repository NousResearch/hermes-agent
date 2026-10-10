import type { TranslationOverrides } from './define-locale'

// Shell notices (remote-display toast, butterbar), spread into ja.ts.
export const jaNotices = {
  remoteDisplayBanner: {
    message: reason =>
      `ソフトウェアレンダリングが有効です — リモートディスプレイを検出しました（${reason}）。ちらつきを防ぐため GPU アクセラレーションは無効化されています。`
  },
  previewDraft: {
    discardTitle: '保存していない変更を破棄しますか？',
    discardBody: label => `${label} には保存していない変更があります。タブを閉じると失われます。`,
    discardConfirm: '変更を破棄'
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
  }
} satisfies Pick<TranslationOverrides, 'remoteDisplayBanner' | 'previewDraft' | 'butterbar'>
