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
  restarting: 'Hermes перезапускается...',
  askRestart: 'Попросить Hermes перезапустить сервер',
  lookingRestart: taskId => `Hermes ищет сервер предпросмотра для перезапуска (${taskId})`,
  restartingTitle: 'Перезапуск сервера предпросмотра',
  restartingMessage: 'Hermes работает в фоне. Следите за прогрессом в консоли предпросмотра.',
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
  blankPageBody: 'Введите адрес выше, чтобы просматривать, или попросите Hermes открыть страницу.',
  finishedRestarting: message => `Hermes завершил перезапуск сервера предпросмотра${message ? `: ${message}` : ''}`,
  failedRestarting: message => `Перезапуск сервера не удался: ${message}`,
  unknownError: 'неизвестная ошибка',
  restartedTitle: 'Сервер предпросмотра перезапущен',
  reloadingNow: 'Перезагружаем предпросмотр.',
  restartFailedTitle: 'Перезапуск предпросмотра не удался',
  restartFailedMessage: 'Hermes не смог перезапустить сервер.',
  stillWorking:
    'Hermes всё ещё работает, но результата перезапуска пока нет. Команда сервера может выполняться в foreground.',
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
  fallbackTitle: 'Предпросмотр'
}
