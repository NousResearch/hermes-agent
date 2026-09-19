import type { TranslationOverrides } from './define-locale'

export const jaCommon = {
  common: {
    apply: '適用',
    back: '戻る',
    save: '保存',
    saving: '保存中…',
    cancel: 'キャンセル',
    change: '変更',
    choose: '選択',
    clear: 'クリア',
    close: '閉じる',
    collapse: '折りたたむ',
    confirm: '確認',
    connect: '接続',
    connecting: '接続中',
    continue: '続ける',
    copied: 'コピーしました',
    copy: 'コピー',
    copyFailed: 'コピーに失敗しました',
    delete: '削除',
    docs: 'ドキュメント',
    done: '完了',
    error: 'エラー',
    expand: '展開',
    failed: '失敗',
    formatJson: 'JSON を整形',
    free: '無料',
    loading: '読み込み中…',
    notSet: '未設定',
    refresh: '更新',
    remove: '削除',
    replace: '置き換え',
    retry: '再試行',
    run: '実行',
    send: '送信',
    set: '設定',
    skip: 'スキップ',
    update: '更新',
    tryHint: term => `「${term}」を試す`,
    on: 'オン',
    off: 'オフ'
  },

  billingBlock: {
    titleNous: 'Nous クレジットが不足しています',
    titleProvider: provider => `クレジット不足 — ${provider}`,
    fallbackMessage: 'アカウントのクレジットが不足しています。続行するにはクレジットを追加してください。',
    openBilling: '請求を開く',
    addCredits: 'クレジットを追加',
    dismiss: '閉じる'
  },

  ui: {
    search: {
      clear: '検索をクリア'
    },
    pagination: {
      label: 'ページング',
      previous: '前へ',
      previousAria: '前のページへ',
      next: '次へ',
      nextAria: '次のページへ'
    },
    sidebar: {
      title: 'サイドバー',
      description: 'モバイルサイドバーを表示します。',
      toggle: open => `サイドバーを${open ? '表示' : '非表示'}`
    }
  }
} satisfies Pick<TranslationOverrides, 'common' | 'billingBlock' | 'ui'>
