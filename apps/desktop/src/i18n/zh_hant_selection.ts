import type { TranslationOverrides } from './define-locale'

export const zh_hantSelectionTranslate = {
  title: '翻譯',
  providerNote: '使用您設定的 Hermes 模型。所選文字可能會透過該提供者離開此裝置。',
  target: '偏好語言',
  preferredHint: '將儲存供後續翻譯使用。如果文字已符合非英文目標語言，Hermes 會改為翻譯成英文。',
  searchLanguages: '搜尋語言…',
  noLanguages: '找不到語言。',
  useLanguageTag: (name, tag) => `使用 ${name}（${tag}）`,
  languageTagHint: '也可以輸入語言標籤，例如 pt-BR 或 zh-Hant。',
  source: '所選文字',
  translation: '翻譯',
  translating: '翻譯中…',
  failed: '翻譯失敗',
  emptyResult: '提供者傳回了空白翻譯。',
  tooLong: '請選取 4,000 個字元以內的文字進行翻譯。',
  retry: '重試',
  copy: '複製',
  copied: '已複製翻譯',
  copyFailed: '無法複製翻譯'
} satisfies TranslationOverrides['selectionTranslate']

export const zh_hantSelectionActions = { readAloud: '朗讀', lookUp: '查詢', translate: '翻譯…', stop: '停止' }
