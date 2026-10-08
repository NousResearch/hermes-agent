import type { LanguageTranslations } from './types_language'

export const jaLanguage = {
  label: '言語',
  description: 'デスクトップインターフェイスの言語を選択します。',
  saving: '言語を保存中…',
  saveError: '言語の更新に失敗しました',
  switchTo: '言語を切り替え',
  searchPlaceholder: '言語を検索…',
  noResults: '言語が見つかりません'
} satisfies Partial<LanguageTranslations>
