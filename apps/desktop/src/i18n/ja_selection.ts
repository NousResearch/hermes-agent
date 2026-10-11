import type { TranslationOverrides } from './define-locale'

export const jaSelectionTranslate = {
  title: '翻訳',
  providerNote:
    '設定済みの Hermes モデルを使用します。選択したテキストは、そのプロバイダーを経由してこのデバイスの外部に送信される場合があります。',
  target: '優先言語',
  preferredHint:
    '今後の翻訳用に保存されます。テキストが英語以外の対象言語と一致する場合、Hermes は代わりに英語へ翻訳します。',
  searchLanguages: '言語を検索…',
  noLanguages: '言語が見つかりません。',
  useLanguageTag: (name, tag) => `${name}（${tag}）を使用`,
  languageTagHint: 'pt-BR や zh-Hant などの言語タグも入力できます。',
  source: '選択したテキスト',
  translation: '翻訳',
  translating: '翻訳中…',
  failed: '翻訳に失敗しました',
  emptyResult: 'プロバイダーから空の翻訳が返されました。',
  tooLong: '翻訳するテキストは4,000文字以内で選択してください。',
  retry: '再試行',
  copy: 'コピー',
  copied: '翻訳をコピーしました',
  copyFailed: '翻訳をコピーできませんでした'
} satisfies TranslationOverrides['selectionTranslate']

export const jaSelectionActions = { readAloud: '読み上げ', lookUp: '調べる', translate: '翻訳…', stop: '停止' }

export const jaContextMenu = {
  link: {
    openInApp: 'アプリ内ブラウザーで開く',
    openExternal: '外部ブラウザーで開く',
    copyUrl: 'URL をコピー',
    copyResolvedUrl: '解決後の URL をコピー'
  },
  image: {
    copyImage: '画像をコピー',
    copyImageAddress: '画像アドレスをコピー',
    saveImageAs: '画像を名前を付けて保存…'
  },
  edit: {
    cut: '切り取り',
    paste: '貼り付け',
    selectAll: 'すべて選択',
    addToDictionary: '辞書に追加'
  },
  page: {
    copyPageUrl: 'ページの URL をコピー',
    inspectElement: '要素を調査'
  }
} satisfies TranslationOverrides['contextMenu']
