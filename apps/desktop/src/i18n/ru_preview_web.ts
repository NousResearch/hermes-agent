import type { TranslationOverride } from '@hermes/shared/i18n'

import { RU_NOUN } from './ru_plural'
import type { Translations } from './types'

// The in-app browser pane's copy (`preview.web`), composed by ru.ts.
export const ruPreviewWeb: TranslationOverride<Translations['preview']['web']> = {
  embeddedPreviewHint:
    'Некоторые сайты блокируют встроенный предпросмотр. Откройте исходную страницу во вкладке браузера.',
  appFailedToBoot: 'Приложение предпросмотра не запустилось',
  serverNotFound: 'Сервер не найден',
  remoteLoopback:
    'Этот адрес указывает на машину, на которой работает ваш агент, а не на эту. Панель браузера загружает страницы локально, поэтому для удалённого dev-сервера нужен порт-форвардинг или доступный hostname.',
  failedToLoad: 'Не удалось загрузить предпросмотр',
  tryAgain: 'Попробовать снова',
  restarting: 'Агент перезапускает сервер...',
  askRestart: 'Попросить агента перезапустить сервер',
  lookingRestart: taskId => `Агент ищет сервер предпросмотра для перезапуска (${taskId})`,
  restartingTitle: 'Перезапуск сервера предпросмотра',
  restartingMessage: 'Агент работает в фоне. Следите за прогрессом в консоли предпросмотра.',
  startRestartFailed: message => `Не удалось запустить перезапуск сервера: ${message}`,
  restartFailed: 'Перезапуск сервера не удался',
  hideConsole: 'Скрыть консоль предпросмотра',
  showConsole: 'Показать консоль предпросмотра',
  hideDevTools: 'Скрыть DevTools предпросмотра',
  openDevTools: 'Открыть DevTools предпросмотра',
  goBack: 'Назад',
  goForward: 'Вперёд',
  reload: 'Перезагрузить страницу',
  address: 'Адрес',
  addressPlaceholder: 'Введите адрес',
  blankPageBody: 'Введите адрес выше, чтобы просматривать, или попросите агента открыть страницу.',
  finishedRestarting: message => `Агент завершил перезапуск сервера предпросмотра${message ? `: ${message}` : ''}`,
  failedRestarting: message => `Перезапуск сервера не удался: ${message}`,
  unknownError: 'неизвестная ошибка',
  restartedTitle: 'Сервер предпросмотра перезапущен',
  reloadingNow: 'Перезагружаем предпросмотр.',
  restartFailedTitle: 'Перезапуск предпросмотра не удался',
  restartFailedMessage: 'Агент не смог перезапустить сервер.',
  stillWorking:
    'Агент всё ещё работает, но результата перезапуска пока нет. Команда сервера может выполняться в foreground.',
  workspaceReloading: 'Рабочее пространство изменилось, перезагружаем предпросмотр',
  fileChanged: url => `Файл изменился, перезагружаем предпросмотр: ${url}`,
  filesChanged: (count, url) =>
    `${count} ${RU_NOUN(count, 'изменение файла', 'изменения файла', 'изменений файла')}, перезагружаем предпросмотр: ${url}`,
  watchFailed: message => `Не удалось отслеживать файл предпросмотра: ${message}`,
  moduleMimeDescription:
    'Модульные скрипты раздаются с неверным MIME-типом. Обычно это значит, что статический файл-сервер раздаёт Vite/React-приложение вместо dev-сервера проекта.',
  loadFailedConsole: (code, message) => `Не удалось загрузить${code ? ` (${code})` : ''}: ${message}`,
  unreachableDescription: 'Страница предпросмотра недоступна.',
  openTarget: url => `Открыть ${url}`,
  fallbackTitle: 'Предпросмотр',
  vaultFill: 'Подставить сохранённый вход',
  vaultTitle: site => `Сохранённые входы для ${site}`,
  vaultLoading: 'Загрузка сохранённых входов…',
  vaultEmpty: 'Для этого сайта нет сохранённого входа',
  vaultUnlock: name => `Разблокировать ${name}`,
  vaultOpenSettings: 'Открыть настройки хранилища учётных данных',
  vaultFilled: label => `${label}: данные подставлены. Проверьте форму и войдите.`,
  vaultFillFailed: 'Не удалось подставить вход',
  vaultLoadFailed: 'Не удалось загрузить сохранённые входы',
  vaultPageNotReady: 'Страница ещё загружается. Повторите, когда загрузка завершится.'
}
