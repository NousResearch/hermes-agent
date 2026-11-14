import type { TranslationOverrides } from './define-locale'
import { RU_NOUN } from './ru_plural'

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
  installDescription: 'Навык будет доступен в новых сеансах. Устанавливайте только из источников, которым доверяете.',
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
  installHint: 'Проверьте исходный код перед установкой. Изменения применяются к новым сеансам.',
  results: (count: number) =>
    `${count.toLocaleString('ru')} ${resultWord(count)}`,
  back: 'Назад к результатам',
  tier: 'Уровень',
  previousSlide: 'Предыдущая рекомендация',
  nextSlide: 'Следующая рекомендация',
  slideOf: (index, count) => `${index} из ${count}`,
  tierLabels: {
    official: 'Официальный',
    verified: 'Проверенный',
    community: 'Сообщество'
  },
  tiers: {
    official: {
      title: 'Официальные плагины',
      blurb: 'Создаются и поддерживаются Nous Research.'
    },
    verified: {
      title: 'Проверенные плагины',
      blurb: 'От компании, стоящей за интеграцией, проверены Nous.'
    }
  },
  moreToExplore: 'Что ещё посмотреть',
  moreToExploreBlurb: 'Небольшие уголки каталога, собранные в одном месте.',
  skillsFor: category => `Навыки из категории «${category}».`,
  discoverKind: kind => `Откройте для себя ${kind.toLowerCase()}`,
  developer: {
    heading: 'Режим разработчика',
    title: 'Создавайте для Hermes',
    pitch: 'Сделали инструмент, коннектор, плагин или мод?',
    pitchFollowUp: 'Хотите, но не знаете как?',
    submit: 'Отправить мод',
    docs: 'Документация для разработчиков'
  },
  toolsets: {
    mostUsed: 'Чаще всего',
    mostUsedBlurb: 'Наборы инструментов, к которым Hermes чаще всего обращается в этом профиле.',
    essentials: 'Основное',
    essentialsBlurb: 'То, на чём держится большая часть работы Hermes.',
    toolCount: count => `${count} ${RU_NOUN(count, 'инструмент', 'инструмента', 'инструментов')}`,
    callCount: (count, calls) => `${count} ${RU_NOUN(calls, 'вызов', 'вызова', 'вызовов')}`,
    themes: {
      work: {
        label: 'Работать в браузере и автоматизировать',
        blurb: 'Управление браузером и рабочим столом, правка файлов, запуск кода и планирование задач.'
      },
      create: {
        label: 'Видеть, слышать и создавать',
        blurb: 'Просмотр изображений и видео, расшифровка речи и генерация медиа.'
      },
      recall: {
        label: 'Память и знания',
        blurb: 'Что Hermes помнит, а также навыки и история, по которым он может искать.'
      },
      connect: {
        label: 'Подключаться и общаться',
        blurb: 'Аккаунты, другие агенты, умный дом и поиск за пределами веба.'
      },
      more: {
        label: 'Другие инструменты',
        blurb: 'Всё остальное, что можно включить в этом профиле.'
      }
    }
  }
} satisfies TranslationOverrides['catalog']

export const ruSkillDeepLink = {
  installTitle: (name: string) => `Установить «${name}»?`,
  installDescription: 'Навык будет доступен в новых сеансах. Устанавливайте только из источников, которым доверяете.',
  installTo: 'Установить в',
  thisComputer: 'Этот компьютер',
  installing: 'Установка…',
  installComplete: (name: string) => `«${name}» установлен`,
  destinationChanged: 'Место установки изменилось. Закройте диалог и снова откройте ссылку установки.',
  installed: 'Установлено',
  source: 'Источник'
} satisfies TranslationOverrides['skillDeepLink']
