import type { MemoryDiscoveryTranslations } from './types_memory_discovery'

export const ruMemoryDiscovery = {
  installed: 'Установленные',
  availableToInstall: 'Доступны для установки',
  installationRequired: 'Требуется установка',
  reviewInstall: 'Проверить и установить',
  exploreAll: 'Посмотреть все…',
  missing: 'Отсутствует',
  installConsent:
    'Устанавливает и включает плагин с зависимостями. Активный провайдер памяти не изменится до явного выбора.',
  builtin: 'Встроенная',
  configureElsewhere: 'Настройте провайдер через CLI или обновите Hermes для сохранения без активации.',
  notReady: 'Завершите настройку и установите зависимости. После установки перезапустите бэкенд и повторите попытку.',
  useFailed: 'Не удалось использовать провайдер. Проверьте настройки и повторите попытку.',

  activeProvider: name => `Активен: ${name}`,
  useProvider: 'Использовать',
  loadFailed: 'Не удалось загрузить провайдеров памяти',
  ownerChanged: 'Вернитесь к подключению и профилю, в которых вы открыли установщик, и повторите попытку.',
  notDiscovered:
    'Пакет установлен, но провайдер памяти ещё не обнаружен. Вернитесь в настройки памяти для повторного поиска.',
  installedNotice: 'Провайдер обнаружен. Сначала настройте его, затем явно выберите для использования.',
  backToMemory: 'К настройкам памяти',
  connect: 'Подключить',
  reconnect: 'Переподключить',
  connectOAuth: 'Подключить через OAuth',
  apiKeySet: 'API-ключ задан',
  oauthSet: 'OAuth подключён',
  waitingConsent: 'Ожидание согласия в браузере…',
  stopWaiting: 'Не ждать',
  stoppedWaiting: 'Ожидание остановлено. Авторизация может ещё выполняться.',
  startFailed: 'Не удалось начать подключение.',
  connectionFailed: 'Не удалось подключиться.'
} satisfies Partial<MemoryDiscoveryTranslations>
