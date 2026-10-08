import type { LanguageTranslations } from './types_language'

export const zhLanguage = {
  label: '语言',
  description: '选择桌面界面的语言。',
  saving: '正在保存语言…',
  saveError: '语言更新失败',
  switchTo: '切换语言',
  searchPlaceholder: '搜索语言…',
  noResults: '未找到语言'
} satisfies Partial<LanguageTranslations>
