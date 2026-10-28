import type { TranslationOverrides } from './define-locale'

// Temporary (incognito) chat copy, composed by ja.ts.
export const jaTemporaryChat = {
  temporaryChat: {
    title: '一時チャット',
    description: '一時チャットは履歴に表示されず、メモリの参照や更新も行いません。チャットを離れると削除されます。',
    placeholder: '一時チャットで質問する',
    use: '一時チャットを使用',
    exit: '一時チャットを終了',
    badge: '一時',
    badgeHint: '保存されません。閉じたとき、または離れている間に 10 分間操作がないと終了します。',
    newChat: '新しい一時チャット',
    unsupportedBackend:
      'この Hermes バックエンドはまだ一時チャットに対応していません。更新するか、一時モードをオフにして通常どおりチャットしてください。',
    endChat: '一時チャットを終了',
    endTitle: 'この一時チャットを終了しますか？',
    endDescription: '完全に削除され、何も保存されません。',
    endConfirm: '終了して削除',
    returnTo: '一時チャットに戻る',
    running: '応答中…'
  }
} satisfies TranslationOverrides
