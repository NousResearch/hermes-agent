import type { TranslationOverrides } from './define-locale'

type RuPlural = (count: number | string, one: string, few: string, many: string) => string

export const ruToolsets = (
  plural: RuPlural
): NonNullable<TranslationOverrides['settings']>['toolsets'] => ({
  loadingConfig: 'Загрузка конфигурации',
  savedTitle: 'Учётные данные сохранены',
  savedMessage: key => `${key} обновлён.`,
  removedTitle: 'Учётные данные удалены',
  removedMessage: key => `${key} удалён.`,
  failedSave: key => `Не удалось сохранить ${key}`,
  failedRemove: key => `Не удалось удалить ${key}`,
  failedReveal: key => `Не удалось показать ${key}`,
  removeConfirm: key => `Удалить ${key} из .env?`,
  set: 'Установить',
  notSet: 'Не задано',
  selectedTitle: 'Провайдер выбран',
  selectedMessage: provider => `Сейчас активен ${provider}.`,
  failedSelect: provider => `Не удалось выбрать ${provider}`,
  failedLoad: 'Не удалось загрузить конфигурацию инструментов',
  noProviderOptions:
    'У этого набора инструментов нет вариантов провайдера — включите его, и он будет работать с вашей текущей конфигурацией.',
  noProviders: 'Для этого набора инструментов сейчас нет доступных провайдеров.',
  ready: 'Готово',
  needsSignIn: 'Нужен вход',
  needsSetup: 'Требуется настройка',
  activeBackend: 'Активен',
  browserAxesHint:
    'Browser Use выбирает драйвер браузера. Провайдер определяет, где запускается браузер. Оба параметра могут быть активны одновременно.',
  activeBackendHint: 'Это ваш активный бэкенд',
  useBackend: 'Использовать этот бэкенд',
  nousIncluded: 'Входит в подписку Nous — войдите в Nous Portal, чтобы активировать.',
  nousAuthNeededTitle: 'Войдите в Nous Portal',
  nousAuthNeededMessage: provider => `${provider} сохранён, но не активируется, пока вы не войдёте в Nous Portal.`,
  nousAuthSignIn: 'Войти',
  nousAuthDoneTitle: 'Nous Portal подключён',
  nousAuthDoneMessage: 'Ваши бэкенды по подписке теперь активны.',
  nousAuthFailed: 'Вход в Nous Portal не завершён',
  noApiKeyRequired: 'API-ключ не требуется.',
  postSetupHint: step =>
    `Этому бэкенду нужна однократная установка (${step}). Выполняется на этой машине — может занять несколько минут.`,
  postSetupInstalledHint: 'Установлено. Повторите настройку, только если что-то сломано.',
  postSetupRun: 'Запустить настройку',
  postSetupRerun: 'Повторить настройку',
  postSetupInstalled: 'Установлено',
  postSetupRunning: 'Установка…',
  postSetupStarting: 'Запуск…',
  postSetupCompleteTitle: 'Настройка завершена',
  postSetupCompleteMessage: step => `${step} установлен.`,
  postSetupErrorTitle: 'Настройка завершилась с ошибками',
  postSetupErrorMessage: step => `Посмотрите журнал ${step}.`,
  postSetupFailed: step => `Не удалось выполнить настройку ${step}`,
  webSearchActive: backend => `Поиск: ${backend}`,
  webExtractActive: backend => `Извлечение: ${backend}`,
  webCapabilityUnset: 'не задано',
  webUseForSearch: 'Использовать для поиска',
  webUseForExtract: 'Использовать для извлечения',
  webUsedForSearch: 'Бэкенд поиска',
  webUsedForExtract: 'Бэкенд извлечения',
  webCapabilitySelectedMessage: (provider, capability) => `${provider} теперь отвечает за веб-${capability}.`,
  failedSelectCapability: provider => `Не удалось настроить ${provider}`,
  loadingModels: 'Загрузка каталога моделей…',
  modelSectionTitle: 'Модель',
  modelCount: count => `${count} ${plural(count, 'модель', 'модели', 'моделей')}`,
  modelInUse: 'В использовании',
  modelDefault: 'по умолчанию',
  modelInactiveHint: 'Сначала выберите этот бэкенд, чтобы изменить его модель.',
  modelSelectedTitle: 'Модель выбрана',
  modelSelectedMessage: model => `${model} применится к новым сеансам.`,
  failedSelectModel: model => `Не удалось выбрать ${model}`,
  terminalBackend: {
    sectionTitle: 'Бэкенд выполнения',
    loading: 'Проверка бэкендов выполнения…',
    failedLoad: 'Не удалось загрузить бэкенды терминала',
    ready: 'Готово',
    needsSetup: 'Нужна настройка',
    unavailable: 'Недоступен',
    inUse: 'В использовании',
    selectedTitle: 'Бэкенд выбран',
    selectedMessage: backend =>
      `Команды терминала теперь выполняются через ${backend}. Применится к новым сеансам.`,
    failedSelect: backend => `Не удалось выбрать ${backend}`,
    needsSetupHint:
      'Этот бэкенд уже выбран без завершённой настройки — команды будут завершаться ошибкой, пока настройка не будет завершена.',
    needsSetupConfirmTitle: backend => `Всё равно выбрать ${backend}?`,
    needsSetupConfirmDescription: detail =>
      `${detail} Сеансы, запущенные после этого изменения, останутся без терминала и файловых инструментов, пока настройка не завершена.`,
    needsSetupConfirmDescriptionGeneric:
      'Этот бэкенд ещё не настроен. Сеансы, запущенные после этого изменения, останутся без терминала и файловых инструментов, пока настройка не завершена.',
    needsSetupConfirmAction: 'Выбрать всё равно'
  }
})
