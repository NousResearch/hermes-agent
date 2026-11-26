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
      backgroundExited: 'Служба, в которой работают ваши чаты, неожиданно закрылась. Перезапустите её, чтобы продолжить, — ваши чаты и настройки в сохранности.',
      backgroundExitedDuringStartup: 'Фоновый процесс Hermes завершился при запуске.',
      backendStopped: 'Hermes перестал работать в фоне',
      restartHermes: 'Перезапустить Hermes',
      openLogs: 'Открыть журналы',
      desktopBootFailed: 'Не удалось запустить приложение',
      gatewayConnectionLost: 'Hermes потерял соединение',
      gatewayConnectionLostDetail: 'Продолжаем попытки переподключиться. Можно и дальше читать и писать черновики. Если это не прекратится, переподключитесь сейчас или проверьте настройки подключения.',
      reconnectNow: 'Переподключиться сейчас',
      connectionSettings: 'Настройки подключения',
      gatewaySignInRequired: 'Удалённый Hermes завершил ваш сеанс',
      gatewaySignInRequiredDetail: 'Войдите снова, чтобы восстановить соединение. Ваши чаты и настройки сохранены.',
      signInAgain: 'Войти снова',
      ipcBridgeUnavailable: 'Hermes Desktop не может связаться со своим фоновым слоем. Перезапустите приложение.'
    },
    causes: {
      exitedEarly: 'Фоновая служба Hermes остановилась сразу после запуска.',
      timedOut: 'Фоновая служба Hermes не ответила вовремя.',
      permission: 'Hermes не смог записать данные в свою папку из-за недостатка прав.',
      diskFull: 'Диск заполнен, поэтому Hermes не смог запуститься.',
      portInUse: 'Нужный Hermes сетевой порт занят другой программой.',
      installMissing: 'Не хватает части файлов Hermes. Нажмите «Восстановить установку», чтобы вернуть их.'
    },
    failure: {
      title: 'Hermes не удалось запустить',
      description:
        'Фоновая служба Hermes не запустилась. Попробуйте один из шагов восстановления ниже. Ничто из этого не удаляет ваши чаты и настройки.',
      details: 'Подробности',
      remoteTitle: 'Требуется вход в удалённый шлюз',
      remoteDescription:
        'Сеанс удалённого шлюза истёк. Войдите снова, чтобы переподключиться. Ничто из этого не удаляет ваши чаты и настройки.',
      webappTitle: 'Войдите в Hermes',
      webappDescription: 'Запустите hermes webapp на этом компьютере, чтобы войти в этом браузере.',
      retry: 'Повторить',
      repairInstall: 'Восстановить установку',
      useLocalGateway: 'Использовать локальный шлюз',
      gatewaySettings: 'Настройки шлюза',
      back: 'Назад',
      openLogs: 'Открыть журналы',
      repairHint: 'Восстановление перезапускает установщик — на чистой машине это может занять несколько минут.',
      bundledReinstallHint: 'Эту встроенную установку нельзя восстановить из приложения — переустановите приложение, чтобы вернуть его бэкенд.',
      reinstallApp: 'Переустановить приложение',
      remoteSignInHint: signInLabel => `Выход из сохранённого сеанса удалённого браузера, затем открытие ${signInLabel}. Чтобы перейти на встроенный бэкенд, используйте локальный шлюз.`,
      signOutAndSignIn: 'Выйти и войти',
      remoteFailureHint: 'Проверьте URL шлюза и вход в настройках шлюза или переключитесь на локальный шлюз.',
      cloudDownTitle: 'Агент Nous Cloud недоступен',
      cloudDownDescription: 'Облачный агент Nous, к которому подключается этот шлюз, отвечает ошибкой сервера. Перезапустить его отсюда нельзя — проверьте его статус, переключитесь на локальный шлюз или обратитесь в поддержку.',
      cloudDownHint: 'Кнопки ниже открывают Nous Portal (состояние экземпляра и управление им) и наш Discord для поддержки.',
      cloudDownCheckPortal: 'Проверить статус Portal',
      cloudDownDiscord: 'Помощь в Discord',
      hideRecentLogs: 'Скрыть недавние журналы',
      showRecentLogs: 'Показать недавние журналы',
      signedInTitle: 'Вы вошли в систему',
      signedInMessage: 'Переподключение к удалённому шлюзу…',
      signInIncompleteTitle: 'Вход не завершён',
      signInIncompleteMessage: 'Окно входа закрылось до завершения аутентификации.',
      signInFailed: 'Не удалось войти',
      signInToRemoteGateway: 'Войти в удалённый шлюз',
      signInWithProvider: provider => `Войти через ${provider}`,
      identityProvider: 'вашего провайдера аутентификации'
    },
    updateHold: {
      title: 'Предыдущее обновление всё ещё блокирует Hermes',
      titleUnverified: 'Hermes не может подтвердить завершение последнего обновления',
      description: 'Hermes откладывает запуск, чтобы не загрузить файлы, которые обновление ещё может изменять. Как только блокировка снимется, запуск продолжится автоматически.',
      heldByProcess: pid =>
        `Обновление (процесс ${pid}) завершилось, но запущенный им процесс всё ещё удерживает блокировку установки Hermes.`,
      heldUnknown: 'Обновление завершилось, но запущенный им процесс всё ещё удерживает блокировку установки Hermes.',
      unverified: 'Помощнику обновления не удалось определить, какой процесс сейчас удерживает блокировку установки Hermes. Проверки продолжаются.',
      since: time => `Ожидание с ${time}`,
      lastChecked: time => `Последняя проверка: ${time}`,
      recoveryHint: 'Обычно блокировка снимается через несколько минут. Если этого не произошло, закройте Hermes, завершите оставшиеся процессы git или hermes (либо перезагрузите компьютер), затем снова откройте Hermes.',
      checkAgain: 'Проверить снова',
      quit: 'Выйти из Hermes',
      openLogs: 'Открыть журналы',
      startAnyway: 'Всё равно запустить…',
      confirmTitle: 'Запустить Hermes, пока обновление удерживает блокировку?',
      confirmBody: 'Оставшийся процесс обновления ещё может изменять файлы Hermes. При запуске сейчас может загрузиться частично обновлённая установка, которая не будет работать до повторного обновления. Hermes запишет ваш выбор в журнал и сохранит маркер обновления.',
      confirmKeepWaiting: 'Продолжить ожидание',
      confirmStart: 'Всё равно запустить',
      startAnywayRefused: 'Процесс, удерживающий блокировку установки, изменился до запуска Hermes. Проверьте сведения о блокировке и попробуйте снова.'
    }
  }
} satisfies Pick<TranslationOverrides, 'boot'>
