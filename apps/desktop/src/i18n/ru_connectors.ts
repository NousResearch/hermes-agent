import type { TranslationOverrides } from './define-locale'

export const ruConnectors = {
  connectors: {
    directory: {
      suggested: 'Рекомендуемые',
      allApps: 'Все приложения',
      receiptConnected: apps => `Подключены: ${apps}`,
      receiptNone: 'Новые приложения не подключены',
      receiptTimedOut: 'Закрыто через 10 минут',
      finishInBrowser: 'Завершите в браузере',
      signInFailed: 'Не удалось подключиться. Нажмите, чтобы повторить'
    }
  },
  sessionImport: {
    title: 'Продолжить из другого приложения',
    subtitle: 'Перенесите разговор в Hermes и продолжите с того места, где остановились.',
    action: 'Импортировать сессию',
    readingFrom: 'Читаем с',
    connectedComputer: 'подключённого компьютера',
    destination: 'Импорт в',
    all: 'Все',
    search: 'Поиск по загруженным сессиям',
    scanning: 'Поиск разговоров',
    scanError: 'Не удалось найти сессии',
    scanHelp: 'Проверьте подключение к серверу и повторите попытку. Старому серверу может требоваться обновление.',
    empty: 'Разговоров пока нет',
    emptyHelp: 'Здесь появятся сессии Claude Code и Codex с этого сервера.',
    noMatches: 'Совпадений нет',
    searchHelp: 'Попробуйте другой заголовок или папку либо загрузите ещё сессии.',
    skipped: 'Некоторые журналы пусты, недоступны или слишком велики для просмотра.',
    more: 'Загрузить ещё сессии',
    messages: 'сообщений',
    choose: 'Разговор, который стоит продолжить',
    chooseHelp: 'Выберите сессию, чтобы прочитать историю перед импортом в Hermes.',
    previewLoading: 'Открываем просмотр',
    previewError: 'Просмотр недоступен',
    previewHelp: 'Исходный файл мог переместиться или измениться. Обновите список и повторите попытку.',
    previewLimit: 'Просмотр сокращён для удобства чтения. Импортируется весь разговор.',
    you: 'Вы',
    snapshot: 'Этот разговор уже есть в Hermes. Откройте существующую копию, чтобы продолжить.',
    copyNotice:
      'Копируется текст разговора. Исходные файлы не меняются. Вывод инструментов и рассуждения не переносятся.',
    importing: 'Импорт…',
    open: 'Открыть в Hermes',
    continue: 'Продолжить в Hermes',
    importError: 'Не удалось импортировать разговор.'
  }
} satisfies Pick<TranslationOverrides, 'connectors' | 'sessionImport'>
