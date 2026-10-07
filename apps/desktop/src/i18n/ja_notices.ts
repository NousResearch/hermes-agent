import type { TranslationOverrides } from './define-locale'

// Shell notices (remote-display toast, butterbar), spread into ja.ts.
export const jaNotices = {
  remoteDisplayBanner: {
    message: reason =>
      `ソフトウェアレンダリングが有効です — リモートディスプレイを検出しました（${reason}）。ちらつきを防ぐため GPU アクセラレーションは無効化されています。`
  },
  butterbar: {
    goTo: (index, total) => `お知らせ ${index} / ${total} を表示`
  }
} satisfies Pick<TranslationOverrides, 'remoteDisplayBanner' | 'butterbar'>
