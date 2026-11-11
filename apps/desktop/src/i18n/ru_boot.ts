import type { TranslationOverrides } from './define-locale'

// The boot screen's copy, composed by ru.ts.
export const ruBoot = {
  boot: {
    ready: 'Hermes Desktop готов',
    desktopBootFailedWithMessage: message => `Не удалось запустить приложение: ${message}`,
    steps: {
      connectingGateway: 'Подключение к компьютеру',
      loadingSettings: 'Загрузка настроек Hermes',
      loadingSessions: 'Загрузка последних сеансов',
      retryingRemoteBackend: 'Переподключение к удалённому бэкенду Hermes…',
      startingDesktopConnection: 'Запуск подключения приложения',
      startingHermesDesktop: 'Запуск Hermes Desktop…'
    },
    errors: {
      backgroundExited: 'Фоновый процесс Hermes завершён.',
      backgroundExitedDuringStartup: 'Фоновый процесс Hermes завершился при запуске.',
      backendStopped: 'Бэкенд остановлен',
      desktopBootFailed: 'Не удалось запустить приложение',
      gatewayConnectionLost: 'Соединение с компьютером потеряно',
      gatewaySignInRequired: 'Требуется вход на компьютер',
      ipcBridgeUnavailable: 'IPC-мост приложения недоступен.'
    },
    failure: {
      title: 'Hermes не удалось запустить',
      description:
        'Фоновая служба Hermes не запустилась. Попробуйте один из шагов восстановления ниже. Ничто из этого не удаляет ваши чаты и настройки.',
      remoteTitle: 'Требуется вход на удалённый компьютер',
      remoteDescription:
        'Сессия на удалённом компьютере истекла. Войдите снова, чтобы переподключиться. Ничто из этого не удаляет ваши чаты и настройки.',
      retry: 'Повторить',
      repairInstall: 'Восстановить установку',
      useLocalGateway: 'Использовать локальный компьютер',
      gatewaySettings: 'Настройки компьютера',
      back: 'Назад',
      openLogs: 'Открыть журналы',
      repairHint: 'Восстановление перезапускает установщик — на чистой машине это может занять несколько минут.',
      remoteSignInHint: signInLabel =>
        `Выход из сохранённой сессии удалённого браузера, затем открытие ${signInLabel}. Чтобы перейти на встроенный бэкенд, используйте локальный компьютер.`,
      signOutAndSignIn: 'Выйти и войти',
      remoteFailureHint: 'Проверьте URL и вход в настройках компьютера или переключитесь на локальный компьютер.',
      hideRecentLogs: 'Скрыть недавние журналы',
      showRecentLogs: 'Показать недавние журналы',
      signedInTitle: 'Вы вошли в систему',
      signedInMessage: 'Переподключение к удалённому компьютеру…',
      signInIncompleteTitle: 'Вход не завершён',
      signInIncompleteMessage: 'Окно входа закрылось до завершения аутентификации.',
      signInFailed: 'Не удалось войти',
      signInToRemoteGateway: 'Войти на удалённый компьютер',
      signInWithProvider: provider => `Войти через ${provider}`,
      identityProvider: 'вашему провайдеру аутентификации'
    }
  }
} satisfies Pick<TranslationOverrides, 'boot'>
