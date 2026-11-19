import type { TranslationOverrides } from './define-locale'

const resultWord = (count: number) => {
  const n = Math.abs(count) % 10
  const nn = Math.abs(count) % 100

  return n === 1 && nn !== 11 ? 'результат' : n >= 2 && n <= 4 && (nn < 12 || nn > 14) ? 'результата' : 'результатов'
}

export const ruCatalog = {
  add: 'Добавить',
  added: 'Добавлено',
  discover: 'Открывайте новое',
  featured: 'Рекомендуемые',
  explorePlugins: 'Обзор плагинов',
  exploreSkills: 'Обзор навыков',
  mostStarred: 'Больше всего звёзд',
  newest: 'Новинки',
  recentlyUpdated: 'Недавно обновлённые',
  mostUsed: 'Часто используемые',
  updateAvailable: 'Обновление',
  updatesAvailable: (count: number) => `Обновления · ${count}`,
  alphabetical: 'Название',
  sortBy: 'Сортировать по',
  seeAll: 'Показать все',
  related: 'Похожие',
  tags: 'Теги',
  screenshots: 'Снимки экрана',
  listView: 'Список',
  cardView: 'Карточки',
  installTitle: (name: string) => `Установить «${name}»?`,
  installDescription: 'Навык будет доступен в новых сессиях. Устанавливайте только из источников, которым доверяете.',
  installTo: 'Установить в',
  thisComputer: 'Этот компьютер',
  installing: 'Установка…',
  installComplete: (name: string) => `«${name}» установлен`,
  destinationChanged: 'Место установки изменилось. Закройте диалог и снова откройте ссылку установки.',
  install: 'Установить',
  installed: 'Установленные',
  searchSkills: 'Поиск навыков',
  searchPlugins: 'Поиск плагинов',
  allSources: 'Все источники',
  allCategories: 'Все категории',
  about: 'Описание',
  author: 'Автор',
  source: 'Источник',
  category: 'Категория',
  version: 'Версия',
  platforms: 'Платформы',
  requires: 'Требования',
  tools: 'Инструменты',
  hooks: 'Хуки',
  middleware: 'Промежуточное ПО',
  commands: 'Команды',
  license: 'Лицензия',
  addedDate: 'Добавлено',
  updatedDate: 'Обновлено',
  repository: 'Репозиторий',
  documentation: 'Документация',
  noResults: 'Совпадений нет',
  tryAnother: 'Попробуйте другой запрос или сбросьте фильтры.',
  clearFilters: 'Сбросить фильтры',
  filters: 'Фильтры',
  loadFailed: 'Не удалось загрузить каталог',
  retry: 'Попробовать снова',
  more: 'Показать ещё',
  pinned: 'Проверенный коммит',
  snapshotHint: 'Из каталога Hermes. При просмотре нет обращений к исходным репозиториям.',
  installHint: 'Проверьте исходный код перед установкой. Изменения применяются к новым сессиям.',
  results: (count: number) =>
    `${count.toLocaleString('ru')} ${resultWord(count)}`,
  back: 'Назад к результатам'
,
  provider: {
    review: (action: string) => `Проверить ${action}`,
    confirm: (action: string) => `Подтвердить ${action}`,
    actions: { install: 'установку', update: 'обновление', uninstall: 'удаление' },
    managed: 'Управляется этим источником. Проверяйте изменения здесь, а не редактируйте и не удаляйте локальные файлы.',
    loading: 'Загрузка источника…',
    empty: 'В этом источнике нет подходящих навыков.',
    signIn: 'Войдите через плагин-поставщик, чтобы просматривать этот источник.',
    unavailable: 'Этот источник недоступен. Ваши локальные навыки не изменены.',
    incomplete: 'Проверка не завершена, или политика поставщика не разрешает подтверждение.',
    changed: 'Эта проверка устарела. Закройте её и запросите новую проверку.',
    unknown: 'Результат пока неизвестен. Восстановите исходное подтверждение, прежде чем начинать другое действие.',
    recover: 'Восстановить исходную операцию',
    finalizing: 'Завершение локального статуса…',
    setup: 'Только инструкции по настройке. Подтверждение не выполняет эти инструкции.',
    more: (source: string) => `Загрузить ещё: ${source}`,
    preview: 'Предпросмотр навыка',
    status: {
      not_installed: 'Не установлен', installed: 'Установлен', removed: 'Удалён вне этого источника',
      unmanaged_local: 'Неуправляемая локальная копия', pending: 'Операция ожидает', unavailable: 'Источник недоступен',
      unknown: 'Результат неизвестен', completed: 'Завершено', failed: 'Операция отклонена'
    }
  }
} satisfies TranslationOverrides['catalog']

export const ruSkillDeepLink = {
  installTitle: (name: string) => `Установить «${name}»?`,
  installDescription: 'Навык будет доступен в новых сессиях. Устанавливайте только из источников, которым доверяете.',
  installTo: 'Установить в',
  thisComputer: 'Этот компьютер',
  installing: 'Установка…',
  installComplete: (name: string) => `«${name}» установлен`,
  destinationChanged: 'Место установки изменилось. Закройте диалог и снова откройте ссылку установки.',
  installed: 'Установленные',
  source: 'Источник'
} satisfies TranslationOverrides['skillDeepLink']
