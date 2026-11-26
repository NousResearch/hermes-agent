import type { TranslationOverrides } from './define-locale'
import { ruBuiltinToolsets } from './ru_builtin_toolsets'

const resultWord = (count: number) => {
  const n = Math.abs(count) % 10
  const nn = Math.abs(count) % 100

  return n === 1 && nn !== 11 ? 'результат' : n >= 2 && n <= 4 && (nn < 12 || nn > 14) ? 'результата' : 'результатов'
}

export const ruCatalog = {
  sourceLabels: { local: 'Локальный', optional: 'Дополнительный' },
  pluginCategories: {
    desktop: { label: "Рабочий стол", blurb: "Панели, вкладки и представления Hermes Desktop." },
    memory: { label: "Память", blurb: "Сервисы памяти и модули управления контекстом." },
    platform: { label: "Платформы", blurb: "Подключение мессенджеров и каналов связи." },
    web: { label: "Интернет и браузер", blurb: "Поиск, извлечение данных со страниц и управление браузером." },
    tools: { label: "Инструменты", blurb: "Дополнительные инструменты, которыми может пользоваться агент." },
    voice: { label: "Голос", blurb: "Распознавание речи, озвучивание текста и звук в реальном времени." },
    automation: { label: "Автоматизация", blurb: "Обработка событий, условия запуска и автоматизация сеансов." },
    models: { label: "Модели", blurb: "Провайдеры моделей и сервисы выполнения запросов." },
    general: { label: "Разное", blurb: "Плагины, которые относятся к нескольким категориям." },
  },
  builtinToolsets: ruBuiltinToolsets,
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
  tierLabels: {
    official: 'Официальный',
    verified: 'Проверенный',
    community: 'Сообщество'
  },
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
