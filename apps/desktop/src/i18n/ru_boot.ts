import type { TranslationOverrides } from './define-locale'

// The boot screen's copy, composed by ru.ts.
export const ruBoot = {
  boot: {
    ready: 'Hermes Desktop готов',
    desktopBootFailedWithMessage: message => `Не удалось запустить приложение: ${message}`,
    steps: {
      connectingGateway: 'Подключение к шлюзу',
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
      gatewayConnectionLost: 'Соединение с шлюзом потеряно',
      gatewaySignInRequired: 'Требуется вход в шлюз',
      ipcBridgeUnavailable: 'IPC-мост приложения недоступен.',
      restartHermes: 'Перезапустить Hermes',
      openLogs: 'Открыть журналы',
      gatewayConnectionLostDetail:
        'Повторяем попытки в фоне. Можно продолжать читать и писать черновики — откройте настройки шлюза, если это не прекратится.',
      reconnectNow: 'Переподключиться сейчас',
      connectionSettings: 'Настройки подключения',
      gatewaySignInRequiredDetail: 'Войдите снова, чтобы восстановить соединение. Ваши чаты и настройки сохранены.',
      signInAgain: 'Войти снова'
    },
    failure: {
      title: 'Hermes не удалось запустить',
      description:
        'Фоновый шлюз не запустился. Попробуйте один из шагов восстановления ниже. Ничто из этого не удаляет ваши чаты и настройки.',
      remoteTitle: 'Требуется вход в удалённый шлюз',
      remoteDescription:
        'Сессия удалённого шлюза истекла. Войдите снова, чтобы переподключиться. Ничто из этого не удаляет ваши чаты и настройки.',
      retry: 'Повторить',
      repairInstall: 'Восстановить установку',
      useLocalGateway: 'Использовать локальный шлюз',
      gatewaySettings: 'Настройки шлюза',
      back: 'Назад',
      openLogs: 'Открыть журналы',
      repairHint: 'Восстановление перезапускает установщик — на чистой машине это может занять несколько минут.',
      remoteSignInHint: signInLabel =>
        `Выход из сохранённой сессии удалённого браузера, затем открытие ${signInLabel}. Чтобы перейти на встроенный бэкенд, используйте локальный шлюз.`,
      signOutAndSignIn: 'Выйти и войти',
      remoteFailureHint: 'Проверьте URL шлюза и вход в настройках шлюза или переключитесь на локальный шлюз.',
      hideRecentLogs: 'Скрыть недавние журналы',
      showRecentLogs: 'Показать недавние журналы',
      signedInTitle: 'Вы вошли в систему',
      signedInMessage: 'Переподключение к удалённому шлюзу…',
      signInIncompleteTitle: 'Вход не завершён',
      signInIncompleteMessage: 'Окно входа закрылось до завершения аутентификации.',
      signInFailed: 'Не удалось войти',
      signInToRemoteGateway: 'Войти в удалённый шлюз',
      signInWithProvider: provider => `Войти через ${provider}`,
      identityProvider: 'вашему провайдеру аутентификации',
      details: 'Подробности',
      bundledReinstallHint:
        'Эту встроенную установку нельзя восстановить из приложения — переустановите приложение, чтобы вернуть его бэкенд.',
      reinstallApp: 'Переустановить приложение',
      cloudDownTitle: 'Агент Nous Cloud недоступен',
      cloudDownDescription:
        'Облачный агент Nous, к которому подключается этот шлюз, отвечает ошибкой сервера. Перезапустить его отсюда нельзя — проверьте его статус, переключитесь на локальный шлюз или обратитесь в поддержку.',
      cloudDownHint:
        'Кнопки ниже открывают Nous Portal (состояние и управление инстансом) и наш Discord для поддержки.',
      cloudDownCheckPortal: 'Проверить статус Portal',
      cloudDownDiscord: 'Помощь в Discord'
    },
    causes: {
      exitedEarly: 'Фоновая служба Hermes остановилась сразу после запуска.',
      timedOut: 'Фоновая служба Hermes не ответила вовремя.',
      permission: 'Hermes не смог записать данные в свою папку из-за недостатка прав.',
      diskFull: 'Диск заполнен, поэтому Hermes не смог запуститься.',
      portInUse: 'Нужный Hermes сетевой порт занят другой программой.',
      installMissing: 'Не хватает части файлов Hermes. Нажмите «Восстановить установку», чтобы вернуть их.'
    },
    updateHold: {
      title: 'Предыдущее обновление всё ещё блокирует Hermes',
      titleUnverified: 'Hermes не может подтвердить завершение последнего обновления',
      description:
        'Hermes откладывает запуск, чтобы не загрузить файлы, которые обновление ещё может изменять. Как только блокировка снимется, запуск продолжится автоматически.',
      heldUnknown: 'Обновление завершилось, но запущенный им процесс всё ещё удерживает блокировку установки Hermes.',
      unverified:
        'Помощнику обновления не удалось определить, какой процесс сейчас удерживает блокировку установки Hermes. Проверки продолжаются.',
      recoveryHint:
        'Обычно блокировка снимается через несколько минут. Если этого не произошло, закройте Hermes, завершите оставшиеся процессы git или hermes (либо перезагрузите компьютер), затем снова откройте Hermes.',
      checkAgain: 'Проверить снова',
      quit: 'Выйти из Hermes',
      openLogs: 'Открыть журналы',
      startAnyway: 'Всё равно запустить…',
      confirmTitle: 'Запустить Hermes, пока обновление удерживает блокировку?',
      confirmBody:
        'Оставшийся процесс обновления ещё может изменять файлы Hermes. При запуске сейчас может загрузиться частично обновлённая установка, которая не будет работать до повторного обновления. Hermes запишет ваш выбор в журнал и сохранит маркер обновления.',
      confirmKeepWaiting: 'Продолжить ожидание',
      confirmStart: 'Всё равно запустить',
      startAnywayRefused:
        'Процесс, удерживающий блокировку установки, изменился до запуска Hermes. Проверьте сведения о блокировке и попробуйте снова.',
      heldByProcess: pid =>
        `Обновление (процесс ${pid}) завершилось, но запущенный им процесс всё ещё удерживает блокировку установки Hermes.`,
      since: time => `Ожидание с ${time}`,
      lastChecked: time => `Последняя проверка: ${time}`
    }
  }
} satisfies Pick<TranslationOverrides, 'boot'>
