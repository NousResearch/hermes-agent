import type { TranslationOverrides } from './define-locale'

export const jaConnectors = {
  sessionImport: {
    title: '別のアプリから続ける',
    subtitle: '会話をHermesに取り込み、続きを始めましょう。',
    action: 'セッションを取り込む',
    readingFrom: '読み込み元',
    connectedComputer: '接続先のコンピューター',
    destination: '取り込み先',
    all: 'すべて',
    search: '読み込み済みのセッションを検索',
    scanning: '会話を検索中',
    scanError: 'セッションを取得できません',
    scanHelp: 'バックエンドの接続を確認して再試行してください。古いバックエンドは更新が必要な場合があります。',
    empty: '会話が見つかりません',
    emptyHelp: 'このバックエンドのClaude CodeとCodexのセッションがここに表示されます。',
    noMatches: '一致する会話がありません',
    searchHelp: '別のタイトルやフォルダーを検索するか、セッションを追加で読み込んでください。',
    skipped: '空、読み込み不可、または大きすぎるログをスキップしました。',
    more: 'さらに読み込む',
    messages: 'メッセージ',
    choose: '会話の続きを始めましょう',
    chooseHelp: 'セッションを選び、取り込む前に履歴を確認できます。',
    previewLoading: 'プレビューを開いています',
    previewError: 'プレビューできません',
    previewHelp: '元のファイルが移動または変更された可能性があります。一覧を更新してください。',
    previewLimit: '読みやすいようにプレビューを省略しています。取り込み時は会話全体をコピーします。',
    you: 'あなた',
    snapshot: 'この会話は取り込み済みです。既存のコピーを開いて続けられます。',
    copyNotice: '会話のテキストをコピーします。元のファイルは変更されません。ツール出力と推論は含まれません。',
    importing: '取り込み中…',
    open: 'Hermesで開く',
    continue: 'Hermesで続ける',
    importError: '会話を取り込めませんでした。'
  }
} satisfies Pick<TranslationOverrides, 'sessionImport'>
