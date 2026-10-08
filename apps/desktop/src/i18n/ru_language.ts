import type { LanguageTranslations } from './types_language'

export const ruLanguage = {
  label: 'Язык',
  description: 'Выберите язык интерфейса приложения.',
  saving: 'Сохранение языка…',
  saveError: 'Не удалось обновить язык',
  switchTo: 'Сменить язык',
  searchPlaceholder: 'Поиск языка…',
  noResults: 'Языки не найдены'
} satisfies Partial<LanguageTranslations>
