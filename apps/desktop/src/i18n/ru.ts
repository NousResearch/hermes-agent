import { defineFieldCopy } from '@/app/settings/field-copy'

import { countRu, pluralRu } from './ru-plural'
import type { Translations } from './types'

export const ru: Translations = {
  common: {
    apply: 'Применить',
    back: 'Назад',
    save: 'Сохранить',
    saving: 'Сохранение…',
    cancel: 'Отмена',
    change: 'Изменить',
    choose: 'Выбрать',
    clear: 'Очистить',
    close: 'Закрыть',
    collapse: 'Свернуть',
    confirm: 'Подтвердить',
    connect: 'Подключить',
    connecting: 'Подключение',
    continue: 'Продолжить',
    copied: 'Скопировано',
    copy: 'Копировать',
    copyFailed: 'Не удалось скопировать',
    delete: 'Удалить',
    docs: 'Документация',
    done: 'Готово',
    error: 'Ошибка',
    expand: 'Развернуть',
    failed: 'Не удалось',
    formatJson: 'Форматировать JSON',
    free: 'Бесплатно',
    loading: 'Загрузка…',
    notSet: 'Не задано',
    refresh: 'Обновить',
    remove: 'Убрать',
    replace: 'Заменить',
    retry: 'Повторить',
    run: 'Запустить',
    send: 'Отправить',
    set: 'Установить',
    skip: 'Пропустить',
    update: 'Обновить',
    tryHint: term => `Попробуйте «${term}»`,
    on: 'Вкл',
    off: 'Выкл'
  },
  fileMenu: {
    revealFinder: 'Показать в Finder',
    revealExplorer: 'Показать в проводнике',
    revealFileManager: 'Открыть папку',
    revealInSidebar: 'Показать в дереве файлов',
    copyPath: 'Копировать путь',
    copyRelativePath: 'Копировать относительный путь',
    rename: 'Переименовать…',
    delete: 'Удалить',
    renameTitle: 'Переименовать',
    renameLabel: 'Новое имя',
    deleteTitle: name => `Удалить ${name}?`,
    deleteBody: 'Файл будет перемещён в корзину — вы можете восстановить его оттуда.',
    pathCopied: 'Путь скопирован'
  },
  boot: {
    ready: 'Hermes Desktop готов к работе',
    desktopBootFailedWithMessage: message => `Сбой запуска десктопа: ${message}`,
    steps: {
      connectingGateway: 'Подключение к шлюзу Hermes Desktop',
      loadingSettings: 'Загрузка настроек Hermes',
      loadingSessions: 'Загрузка недавних сессий',
      startingDesktopConnection: 'Запуск десктоп-подключения',
      startingHermesDesktop: 'Запуск Hermes Desktop…'
    },
    errors: {
      backgroundExited: 'Фоновый процесс Hermes завершился.',
      backgroundExitedDuringStartup: 'Фоновый процесс Hermes завершился во время запуска.',
      backendStopped: 'Бэкенд остановлен',
      desktopBootFailed: 'Сбой запуска десктопа',
      gatewayConnectionLost: 'Потеряно соединение со шлюзом',
      gatewaySignInRequired: 'Требуется вход в шлюз',
      ipcBridgeUnavailable: 'IPC-мост десктопа недоступен.'
    },
    failure: {
      title: 'Hermes не удалось запустить',
      description:
        'Фоновый шлюз не запустился. Попробуйте один из способов восстановления ниже. Это не удалит ваши чаты и настройки.',
      remoteTitle: 'Требуется вход в удалённый шлюз',
      remoteDescription:
        'Сессия удалённого шлюза истекла. Войдите снова, чтобы подключиться. Это не удалит ваши чаты и настройки.',
      retry: 'Повторить',
      repairInstall: 'Восстановить установку',
      useLocalGateway: 'Использовать локальный шлюз',
      gatewaySettings: 'Настройки шлюза',
      back: 'Назад',
      openLogs: 'Открыть логи',
      repairHint: 'Восстановление перезапускает установщик и может занять несколько минут на новой машине.',
      remoteSignInHint: signInLabel =>
        `Будет выполнен выход из сохранённой удалённой сессии браузера, затем откроется ${signInLabel}. Чтобы переключиться на встроенный бэкенд, используйте локальный шлюз.`,
      signOutAndSignIn: 'Выйти и войти снова',
      remoteFailureHint:
        'Проверьте URL шлюза и выполните вход в разделе «Настройки шлюза» либо переключитесь на локальный шлюз.',
      hideRecentLogs: 'Скрыть недавние логи',
      showRecentLogs: 'Показать недавние логи',
      signedInTitle: 'Вход выполнен',
      signedInMessage: 'Подключение к удалённому шлюзу…',
      signInIncompleteTitle: 'Вход не завершён',
      signInIncompleteMessage: 'Окно входа закрылось до завершения аутентификации.',
      signInFailed: 'Вход не удался',
      signInToRemoteGateway: 'Войти в удалённый шлюз',
      signInWithProvider: provider => `Войти через ${provider}`,
      identityProvider: 'ваш провайдер авторизации'
    }
  },
  notifications: {
    region: 'Уведомления',
    hide: 'Скрыть',
    show: 'Показать',
    more: count =>
      `ещё ${countRu(count, {
        one: 'уведомление',
        few: 'уведомления',
        many: 'уведомлений'
      })}`,
    clearAll: 'Очистить все',
    dismiss: 'Закрыть уведомление',
    details: 'Подробнее',
    copyDetail: 'Копировать подробность',
    copyDetailFailed: 'Не удалось скопировать подробность уведомления',
    backendOutOfDateTitle: 'Бэкенд устарел',
    backendOutOfDateMessage:
      'Ваш бэкенд Hermes старше этой сборки десктопа и может работать некорректно. Обновите для синхронизации.',
    installMethodUnsupportedTitle: 'Неподдерживаемый метод установки',
    updateHermes: 'Обновить Hermes',
    updateReadyTitle: 'Обновление готово',
    updateReadyMessage: count =>
      `Доступно ${count} ${pluralRu(count, {
        one: 'новое изменение',
        few: 'новых изменения',
        many: 'новых изменений'
      })}.`,
    seeWhatsNew: 'Что нового',
    errors: {
      elevenLabsNeedsKey: 'Для ElevenLabs STT требуется ELEVENLABS_API_KEY.',
      elevenLabsRejectedKey: 'ElevenLabs отклонил API-ключ (401).',
      gatewayAuthFailed: 'Ошибка аутентификации шлюза — проверьте API_SERVER_KEY.',
      methodNotAllowed:
        'Десктопный бэкенд отклонил запрос (405 Method Not Allowed). Попробуйте перезапустить Hermes Desktop.',
      microphonePermission: 'Доступ к микрофону запрещён.',
      openaiRejectedApiKey: 'OpenAI отклонил API-ключ.',
      openaiRejectedApiKeyWithStatus: status => `OpenAI отклонил API-ключ (${status} invalid_api_key).`,
      openaiTtsNeedsKey: 'Для OpenAI TTS требуется VOICE_TOOLS_OPENAI_KEY или OPENAI_API_KEY.'
    },
    voice: {
      configureSpeechToText: 'Настройте распознавание речи для использования голосового режима.',
      couldNotStartSession: 'Не удалось запустить голосовую сессию',
      microphoneAccessDenied: 'Доступ к микрофону запрещён.',
      microphoneConstraintsUnsupported: 'Ограничения микрофона не поддерживаются этим устройством.',
      microphoneFailed: 'Сбой микрофона',
      microphoneInUse: 'Микрофон уже используется другим приложением.',
      microphonePermissionDenied: 'Доступ к микрофону запрещён.',
      microphoneStartFailed: 'Не удалось начать запись с микрофона.',
      microphoneUnsupported: 'Эта среда не поддерживает запись с микрофона.',
      noMicrophone: 'Микрофон не найден.',
      noSpeechDetected: 'Речь не обнаружена',
      playbackFailed: 'Не удалось воспроизвести голос',
      recordingFailed: 'Не удалось записать голос',
      transcriptionFailed: 'Не удалось распознать речь',
      transcriptionUnavailable: 'Распознавание речи пока недоступно.',
      tryRecordingAgain: 'Попробуйте записать снова.',
      unavailable: 'Голос недоступен'
    },
    native: {
      approvalTitle: 'Требуется подтверждение',
      approveAction: 'Одобрить',
      rejectAction: 'Отклонить',
      inputTitle: 'Требуется ввод',
      inputBody: 'Hermes ждёт вашего ответа.',
      turnDoneTitle: 'Hermes завершил',
      turnDoneBody: 'Ответ готов.',
      turnErrorTitle: 'Сбой шага',
      backgroundDoneTitle: 'Фоновая задача завершена',
      backgroundFailedTitle: 'Фоновая задача не удалась',
      creditsTitle: 'Кредиты'
    }
  },
  remoteDisplayBanner: {
    message: reason =>
      `Программный рендеринг активен — обнаружен удалённый дисплей (${reason}). GPU-ускорение отключено для предотвращения мерцания.`
  },
  billingBlock: {
    titleNous: 'Закончились кредиты Nous',
    titleProvider: provider => `Закончились кредиты — ${provider}`,
    fallbackMessage: 'На вашем счёте закончились кредиты. Пополните баланс, чтобы продолжить.',
    openBilling: 'Открыть биллинг',
    addCredits: 'Пополнить',
    dismiss: 'Скрыть'
  },
  titlebar: {
    hideSidebar: 'Скрыть боковую панель',
    showSidebar: 'Показать боковую панель',
    search: 'Поиск',
    searchTitle: 'Поиск сессий, представлений и действий',
    swapSidebarSides: 'Поменять панели местами',
    swapSidebarSidesTitle: 'Поменять местами панель сессий и обозреватель файлов',
    hideRightSidebar: 'Скрыть правую панель',
    showRightSidebar: 'Показать правую панель',
    muteHaptics: 'Выключить тактильную отдачу',
    unmuteHaptics: 'Включить тактильную отдачу',
    openSettings: 'Открыть настройки',
    openStarmap: 'Открыть граф памяти',
    openKeybinds: 'Горячие клавиши',
    layoutEditor: 'Редактор макета',
    layoutEditorTitle: 'Редактор макета — ⌘-клик сбрасывает макет'
  },
  keybinds: {
    title: 'Горячие клавиши',
    subtitle: open => `Нажмите на сочетание, чтобы переназначить · ${open} снова открывает эту панель.`,
    search: 'Поиск горячих клавиш…',
    rebind: 'Переназначить',
    reset: 'Сбросить по умолчанию',
    resetAll: 'Сбросить все',
    pressKey: 'Нажмите клавишу…',
    set: 'задано',
    conflictWith: label => `Также назначено на «${label}»`,
    categories: {
      composer: 'Ввод сообщения',
      profiles: 'Профили',
      session: 'Сессия',
      navigation: 'Навигация',
      view: 'Вид'
    },
    actions: {
      'keybinds.openPanel': 'Открыть горячие клавиши',
      'nav.commandPalette': 'Открыть палитру команд',
      'nav.commandCenter': 'Открыть командный центр',
      'nav.settings': 'Открыть настройки',
      'nav.profiles': 'Открыть профили',
      'nav.skills': 'Открыть навыки',
      'nav.messaging': 'Открыть мессенджеры',
      'nav.artifacts': 'Открыть артефакты',
      'nav.cron': 'Открыть расписания',
      'nav.agents': 'Открыть агентов',
      'session.new': 'Новая сессия',
      'session.newTab': 'Новая вкладка сессии',
      'session.newWindow': 'Новая сессия в окне',
      'session.next': 'Следующая сессия',
      'session.prev': 'Предыдущая сессия',
      'session.slot.1': 'Перейти к недавней сессии 1',
      'session.slot.2': 'Перейти к недавней сессии 2',
      'session.slot.3': 'Перейти к недавней сессии 3',
      'session.slot.4': 'Перейти к недавней сессии 4',
      'session.slot.5': 'Перейти к недавней сессии 5',
      'session.slot.6': 'Перейти к недавней сессии 6',
      'session.slot.7': 'Перейти к недавней сессии 7',
      'session.slot.8': 'Перейти к недавней сессии 8',
      'session.slot.9': 'Перейти к недавней сессии 9',
      'session.focusSearch': 'Поиск сессий',
      'session.togglePin': 'Закрепить / открепить текущую сессию',
      'workspace.newWorktree': 'Новое рабочее дерево',
      'composer.focus': 'Перейти к полю ввода',
      'composer.modelPicker': 'Открыть выбор модели',
      'composer.voice': 'Начать / остановить голосовой разговор',
      'view.toggleSidebar': 'Скрыть/показать панель сессий',
      'view.toggleRightSidebar': 'Скрыть/показать обозреватель файлов',
      'view.toggleReview': 'Скрыть/показать панель рецензирования',
      'view.showFiles': 'Показать обозреватель файлов',
      'view.showTerminal': 'Скрыть/показать терминал',
      'view.newTerminal': 'Новый терминал',
      'view.nextTerminal': 'Следующий терминал',
      'view.prevTerminal': 'Предыдущий терминал',
      'view.closeTerminal': 'Закрыть терминал',
      'view.terminalSelection': 'Отправить выделение из терминала в поле ввода',
      'view.closeTab': 'Закрыть вкладку',
      'view.reopenTab': 'Повторно открыть закрытую вкладку',
      'view.flipPanes': 'Поменять стороны панелей',
      'view.findInPage': 'Найти на странице',
      'view.findNext': 'Следующее совпадение',
      'view.findPrevious': 'Предыдущее совпадение',
      'appearance.toggleMode': 'Переключить светлый / тёмный режим',
      'profile.default': 'Перейти к профилю по умолчанию',
      'profile.switch.1': 'Перейти к профилю 1',
      'profile.switch.2': 'Перейти к профилю 2',
      'profile.switch.3': 'Перейти к профилю 3',
      'profile.switch.4': 'Перейти к профилю 4',
      'profile.switch.5': 'Перейти к профилю 5',
      'profile.switch.6': 'Перейти к профилю 6',
      'profile.switch.7': 'Перейти к профилю 7',
      'profile.switch.8': 'Перейти к профилю 8',
      'profile.switch.9': 'Перейти к профилю 9',
      'profile.switch.10': 'Перейти к профилю 10',
      'profile.switch.11': 'Перейти к профилю 11',
      'profile.switch.12': 'Перейти к профилю 12',
      'profile.switch.13': 'Перейти к профилю 13',
      'profile.switch.14': 'Перейти к профилю 14',
      'profile.switch.15': 'Перейти к профилю 15',
      'profile.switch.16': 'Перейти к профилю 16',
      'profile.switch.17': 'Перейти к профилю 17',
      'profile.switch.18': 'Перейти к профилю 18',
      'profile.next': 'Следующий профиль',
      'profile.prev': 'Предыдущий профиль',
      'profile.toggleAll': 'Показать все профили',
      'profile.create': 'Создать профиль',
      'composer.send': 'Отправить сообщение',
      'composer.newline': 'Вставить перенос строки',
      'composer.steer': 'Направить текущий запуск',
      'composer.queue': 'Поставить сообщение в очередь',
      'composer.sendQueued': 'Отправить следующий ход из очереди',
      'composer.mention': 'Сослаться на файлы, папки, URL',
      'composer.slash': 'Палитра слэш-команд',
      'composer.help': 'Быстрая справка',
      'composer.history': 'Перебор всплывающих / история',
      'composer.cancel': 'Закрыть всплывающее · отменить запуск'
    }
  },
  findInPage: {
    next: 'Следующее совпадение',
    previous: 'Предыдущее совпадение'
  },
  language: {
    label: 'Язык',
    description: 'Выберите язык интерфейса десктопа.',
    saving: 'Сохранение языка…',
    saveError: 'Не удалось обновить язык',
    switchTo: 'Сменить язык',
    searchPlaceholder: 'Поиск языков…',
    noResults: 'Языки не найдены'
  },
  settings: {
    closeSettings: 'Закрыть настройки',
    exportConfig: 'Экспорт конфигурации',
    importConfig: 'Импорт конфигурации',
    resetToDefaults: 'Сбросить по умолчанию',
    resetConfirm: 'Сбросить все настройки на значения Hermes по умолчанию?',
    exportFailed: 'Не удалось экспортировать',
    resetFailed: 'Не удалось сбросить',
    nav: {
      providers: 'Провайдеры',
      providerAccounts: 'Аккаунты',
      providerApiKeys: 'API-ключи',
      providerCustomEndpoints: 'Пользовательские эндпоинты',
      gateway: 'Шлюз',
      apiKeys: 'Инструменты и ключи',
      keybinds: 'Горячие клавиши',
      keysTools: 'Инструменты',
      keysSettings: 'Настройки',
      mcp: 'MCP',
      archivedChats: 'Архив чатов',
      about: 'О приложении',
      billing: 'Биллинг',
      notifications: 'Уведомления',
      plugins: 'Плагины'
    },
    plugins: {
      title: 'Плагины десктопа',
      blurb:
        'Расширения интерфейса, загружаемые в это приложение: встроенные в сборку или добавленные в папку desktop-plugins (включая плагины, созданные Hermes). Отключение немедленно выгружает плагин, а настройка сохраняется после перезапуска.',
      count: n =>
        pluralRu(n, {
          one: `Установлен ${n} плагин`,
          few: `Установлено ${n} плагина`,
          many: `Установлено ${n} плагинов`
        }),
      openFolder: 'Открыть папку плагинов',
      rescan: 'Пересканировать',
      reveal: 'Показать в файловом менеджере',
      enable: 'Включить',
      disable: 'Отключить',
      failed: 'ошибка',
      empty: 'Плагины десктопа пока не установлены.',
      kinds: {
        bundled: 'встроенный',
        disk: 'на диске',
        runtime: 'динамический'
      }
    },
    notifications: {
      title: 'Уведомления',
      intro:
        'Системные десктопные уведомления, отдельные от внутриприложевых всплывающих. Они локальны для устройства — каждый компьютер хранит свои настройки.',
      enableAll: 'Включить уведомления',
      enableAllDesc: 'Главный переключатель. Выключите, чтобы заглушить все уведомления ниже.',
      focusedHint: 'Уведомления о завершении срабатывают только когда Hermes в фоне.',
      kinds: {
        approval: {
          label: 'Требуется подтверждение',
          description: 'Команда ждёт вашего одобрения или отклонения.'
        },
        input: {
          label: 'Требуется ввод',
          description: 'Hermes задал вопрос или ему нужен пароль или секрет.'
        },
        turnDone: {
          label: 'Ответ готов',
          description: 'Шаг завершён, пока Hermes был в фоне.'
        },
        turnError: {
          label: 'Сбой шага',
          description: 'Шаг завершился с ошибкой.'
        },
        backgroundDone: {
          label: 'Фоновая задача завершена',
          description: 'Фоновая команда терминала завершилась.'
        },
        credits: {
          label: 'Оповещения о кредитах',
          description: 'Доступ к кредитам приостановлен или восстановлен.'
        }
      },
      test: 'Отправить тестовое уведомление',
      testTitle: 'Hermes',
      testBody: 'Уведомления работают.',
      testSent:
        'Тест отправлен. Если ничего не появилось, проверьте разрешения уведомлений ОС и режим «Не беспокоить».',
      testUnsupported: 'Эта система не поддерживает системные уведомления.',
      completionSoundTitle: 'Звук завершения',
      completionSoundDesc: 'Воспроизводится при завершении шага агента. Выберите пресет и прослушайте здесь.',
      completionSoundPreview: 'Прослушать'
    },
    sections: {
      model: 'Модель',
      chat: 'Чат',
      appearance: 'Внешний вид',
      workspace: 'Рабочее пространство',
      safety: 'Безопасность',
      memory: 'Память и контекст',
      voice: 'Голос',
      advanced: 'Дополнительно'
    },
    searchPlaceholder: {
      about: 'О Hermes Desktop',
      config: 'Поиск настроек...',
      gateway: 'Подключение к шлюзу...',
      keys: 'Поиск API-ключей...',
      mcp: 'Поиск MCP-серверов...',
      sessions: 'Поиск архивных сессий...'
    },
    modeOptions: {
      light: {
        label: 'Светлая',
        description: 'Яркие поверхности десктопа'
      },
      dark: {
        label: 'Тёмная',
        description: 'Рабочее пространство с низким бликованием'
      },
      system: {
        label: 'Системная',
        description: 'Следовать оформлению ОС'
      }
    },
    appearance: {
      title: 'Внешний вид',
      intro:
        'Это настройки отображения только для десктопа. Режим управляет яркостью; тема управляет акцентной палитрой и стилем чата.',
      colorMode: 'Цветовой режим',
      colorModeDesc: 'Выберите фиксированный режим или позвольте Hermes следовать настройке системы.',
      toolViewTitle: 'Отображение вызова инструментов',
      toolViewDesc:
        '«Продукт» скрывает необработанные данные вызовов инструментов; «Технический» показывает полный ввод и вывод.',
      uiScaleTitle: 'Масштаб интерфейса',
      uiScaleDesc: (percent: number) =>
        `Масштабирует текст и элементы управления во всём приложении. Cmd/Ctrl с +, - и 0 также работает. Сейчас: ${percent}%.`,
      translucencyTitle: 'Полупрозрачность окна',
      translucencyDesc: 'Видите рабочий стол через всё окно. Только для macOS и Windows.',
      backdropTitle: 'Фон чата',
      backdropDesc: 'Едва заметное изображение статуи на заднем плане чата.',
      embedsTitle: 'Встроенные превью',
      embedsDesc:
        'Богатые превью загружаются со сторонних сайтов (YouTube, X, …). «Спрашивать» показывает заглушку, пока вы не разрешите каждый; «Всегда» загружает автоматически; «Выкл» оставляет простые ссылки.',
      embedsAsk: 'Спрашивать',
      embedsAlways: 'Всегда',
      embedsOff: 'Выкл',
      embedsReset: (count: number) =>
        `Сбросить ${countRu(count, {
          one: 'разрешённый сервис',
          few: 'разрешённых сервиса',
          many: 'разрешённых сервисов'
        })}`,
      product: 'Продукт',
      productDesc: 'Понятные действия инструментов с краткими сводками.',
      technical: 'Технический',
      technicalDesc: 'Включает сырые аргументы/результаты инструментов и низкоуровневые детали.',
      themeTitle: 'Тема',
      themeDesc: 'Только десктопные палитры. Выбранный режим применяется поверх.',
      themeProfileNote: profile => `Сохранено для профиля ${profile} — каждый профиль хранит свою тему.`,
      installTitle: 'Установить из VS Code',
      installDesc:
        'Вставьте идентификатор расширения Marketplace (например, dracula-theme.theme-dracula), чтобы преобразовать его цветовую тему в десктопную палитру.',
      installPlaceholder: 'publisher.extension',
      installButton: 'Установить',
      installing: 'Установка…',
      installError: 'Не удалось установить эту тему.',
      installed: name => `Установлено «${name}».`,
      removeTheme: 'Удалить тему',
      importedBadge: 'Импортировано',
      pet: {
        title: 'Питомец',
        intro:
          'Возьмите анимированного маскота petdex, который парит над приложением и реагирует на действия Hermes — бегает при выполнении инструментов, празднует при успехе, грустит при ошибках.',
        restartHint:
          'Питомцам нужен быстрый перезапуск — запущенное приложение стартовало до добавления этой функции. Закройте и откройте Hermes снова, затем вернитесь сюда.',
        on: 'Вкл',
        off: 'Выкл',
        scaleTitle: 'Размер',
        scaleDesc: 'Изменить размер плавающего маскота. Применяется везде мгновенно.',
        roamTitle: 'Блуждание',
        roamDesc: 'Позволить питомцу самостоятельно перемещаться по окну в простое.',
        chooseTitle: 'Выберите питомца',
        chooseDesc: 'Выбор устанавливает его (если нужно) и делает активным.',
        searchPlaceholder: 'Поиск питомцев…',
        unreachable: 'Не удалось связаться с галереей petdex. Проверьте подключение и откройте эту страницу заново.',
        noMatch: query => `Нет питомцев, соответствующих «${query}».`,
        installedTag: 'установлен',
        generatedTag: 'Сгенерирован',
        countCapped: (cap, total) => `Показано ${cap} из ${total} — введите для фильтрации.`,
        count: n =>
          `${countRu(n, {
            one: 'питомец',
            few: 'питомца',
            many: 'питомцев'
          })}.`,
        uninstall: name => `Удалить ${name}`,
        delete: name => `Стереть ${name}`,
        deleteTitle: name => `Стереть ${name}?`,
        deleteBody: 'Это навсегда удаляет питомца — его нельзя переустановить.',
        deleteConfirm: 'Стереть',
        rename: name => `Переименовать ${name}`,
        renameTitle: 'Переименовать питомца',
        renamePlaceholder: 'Назовите питомца',
        renameSave: 'Сохранить',
        exportPet: name => `Экспортировать ${name}`,
        adoptFailed: slug => `Не удалось установить ${slug}`,
        uninstallFailed: slug => `Не удалось удалить ${slug}`,
        renameFailed: slug => `Не удалось переименовать ${slug}`,
        exportFailed: slug => `Не удалось экспортировать ${slug}`,
        noneAvailable: 'Нет доступных питомцев для включения прямо сейчас.',
        turnOnFailed: 'Не удалось включить питомца.',
        turnOffFailed: 'Не удалось выключить питомца.'
      }
    },
    fieldLabels: defineFieldCopy({
      model: 'Модель по умолчанию',
      modelContextLength: 'Контекстное окно',
      fallbackProviders: 'Резервные модели',
      toolsets: 'Включённые наборы инструментов',
      timezone: 'Часовой пояс',
      display: {
        personality: 'Стиль ассистента',
        showReasoning: 'Блоки рассуждений'
      },
      desktop: {
        repoScanEnabled: 'Автопоиск репозиториев',
        repoScanRoots: 'Корни поиска репозиториев',
        repoScanExcludePaths: 'Исключённые пути'
      },
      agent: {
        maxTurns: 'Максимум шагов агента',
        imageInputMode: 'Вложения изображений',
        apiMaxRetries: 'Повторы API-запросов',
        serviceTier: 'Уровень обслуживания (Service Tier)',
        toolUseEnforcement: 'Принудительное использование инструментов'
      },
      terminal: {
        cwd: 'Рабочая директория',
        backend: 'Бэкенд выполнения',
        timeout: 'Таймаут команд',
        persistentShell: 'Постоянная оболочка',
        envPassthrough: 'Проброс переменных окружения',
        dockerImage: 'Образ Docker',
        singularityImage: 'Образ Singularity',
        modalImage: 'Образ Modal',
        daytonaImage: 'Образ Daytona'
      },
      fileReadMaxChars: 'Лимит чтения файла',
      toolOutput: {
        maxBytes: 'Лимит вывода терминала',
        maxLines: 'Лимит страниц файла',
        maxLineLength: 'Лимит длины строки'
      },
      codeExecution: {
        mode: 'Режим выполнения кода'
      },
      approvals: {
        mode: 'Режим подтверждений',
        timeout: 'Таймаут подтверждения',
        mcpReloadConfirm: 'Подтверждать перезагрузку MCP'
      },
      commandAllowlist: 'Белый список команд',
      security: {
        redactSecrets: 'Скрывать секреты',
        allowPrivateUrls: 'Разрешить приватные URL'
      },
      browser: {
        allowPrivateUrls: 'Приватные URL в браузере',
        autoLocalForPrivateUrls: 'Локальный браузер для приватных URL'
      },
      checkpoints: {
        enabled: 'Контрольные точки файлов',
        maxSnapshots: 'Лимит контрольных точек'
      },
      voice: {
        recordKey: 'Клавиша голосового ввода',
        maxRecordingSeconds: 'Максимальная длина записи',
        autoTts: 'Озвучивать ответы'
      },
      stt: {
        enabled: 'Речь в текст',
        echoTranscripts: 'Показывать расшифровки',
        provider: 'Провайдер речь-в-текст',
        local: {
          model: 'Локальная модель транскрипции',
          language: 'Язык транскрипции'
        },
        openai: {
          model: 'Модель OpenAI STT'
        },
        groq: {
          model: 'Модель Groq STT'
        },
        mistral: {
          model: 'Модель Mistral STT'
        },
        elevenlabs: {
          modelId: 'Модель ElevenLabs STT',
          languageCode: 'Язык ElevenLabs',
          tagAudioEvents: 'Метки аудиособытий',
          diarize: 'Разделение голосов (диаризация)'
        }
      },
      tts: {
        provider: 'Провайдер текст-в-речь',
        edge: {
          voice: 'Голос Edge'
        },
        openai: {
          model: 'Модель OpenAI TTS',
          voice: 'Голос OpenAI'
        },
        elevenlabs: {
          voiceId: 'Голос ElevenLabs',
          modelId: 'Модель ElevenLabs'
        },
        xai: {
          voiceId: 'Голос xAI (Grok)',
          language: 'Язык xAI',
          speed: 'Скорость воспроизведения xAI',
          autoSpeechTags: 'Авто-теги речи xAI',
          optimizeStreamingLatency: 'Оптимизация задержки стриминга xAI',
          sampleRate: 'Частота дискретизации xAI',
          bitRate: 'Битрейт xAI'
        },
        minimax: {
          model: 'Модель MiniMax TTS',
          voiceId: 'Голос MiniMax'
        },
        mistral: {
          model: 'Модель Mistral TTS',
          voiceId: 'Голос Mistral'
        },
        gemini: {
          model: 'Модель Gemini TTS',
          voice: 'Голос Gemini'
        },
        neutts: {
          model: 'Модель NeuTTS',
          device: 'Устройство NeuTTS'
        },
        kittentts: {
          model: 'Модель KittenTTS',
          voice: 'Голос KittenTTS'
        },
        piper: {
          voice: 'Голос Piper'
        },
        deepinfra: {
          model: 'Модель DeepInfra TTS',
          voice: 'Голос DeepInfra'
        }
      },
      memory: {
        memoryEnabled: 'Постоянная память',
        userProfileEnabled: 'Профиль пользователя',
        memoryCharLimit: 'Бюджет памяти',
        userCharLimit: 'Бюджет профиля',
        provider: 'Провайдер памяти'
      },
      context: {
        engine: 'Контекстный движок'
      },
      compression: {
        enabled: 'Автосжатие',
        threshold: 'Порог сжатия',
        targetRatio: 'Целевая степень сжатия',
        protectLastN: 'Защищённые последние сообщения'
      },
      delegation: {
        model: 'Модель субагентов',
        provider: 'Провайдер субагентов',
        maxIterations: 'Лимит шагов субагента',
        maxConcurrentChildren: 'Параллельные субагенты',
        childTimeoutSeconds: 'Таймаут субагента',
        reasoningEffort: 'Глубина рассуждений субагента'
      },
      updates: {
        nonInteractiveLocalChanges: 'Локальные правки при обновлении из приложения'
      }
    }),
    fieldDescriptions: defineFieldCopy({
      model: 'Используется для новых чатов, если в поле ввода не выбрана другая модель.',
      modelContextLength: 'Оставьте 0, чтобы использовать определённое окно выбранной модели.',
      fallbackProviders: 'Резервные записи в формате provider:model на случай сбоя основной модели.',
      display: {
        personality: 'Стиль ассистента по умолчанию для новых сессий.',
        showReasoning: 'Показывать блоки рассуждений, когда бэкенд их предоставляет.'
      },
      desktop: {
        repoScanEnabled: 'Сканировать локальные папки на git-репозитории для раздела «Проекты».',
        repoScanRoots: 'Папки для сканирования. Оставьте пустым для сканирования домашней директории.',
        repoScanExcludePaths: 'Папки и их потомки, пропускаемые при поиске репозиториев.'
      },
      timezone: 'Используется, когда Hermes нужен локальный временной контекст. Пусто — системный пояс.',
      agent: {
        imageInputMode: 'Как вложения изображений отправляются в модель.',
        maxTurns: 'Верхний предел шагов вызова инструментов до остановки запуска.'
      },
      terminal: {
        cwd: 'Папка проекта по умолчанию для инструментов и терминала.',
        persistentShell: 'Сохранять состояние оболочки между командами (если бэкенд поддерживает).',
        envPassthrough: 'Переменные окружения, передаваемые в выполнение инструментов.',
        dockerImage: 'Образ контейнера при бэкенде Docker.',
        singularityImage: 'Образ при бэкенде Singularity.',
        modalImage: 'Образ при бэкенде Modal.',
        daytonaImage: 'Образ при бэкенде Daytona.'
      },
      codeExecution: {
        mode: 'Насколько строго выполнение кода ограничено текущим проектом.'
      },
      fileReadMaxChars: 'Максимум символов, читаемых Hermes за один запрос файла.',
      approvals: {
        mode: 'Как Hermes обрабатывает команды, требующие явного подтверждения.',
        timeout: 'Сколько запросы подтверждения ждут до таймаута.'
      },
      security: {
        redactSecrets: 'По возможности скрывать обнаруженные секреты от модели.'
      },
      checkpoints: {
        enabled: 'Создавать снапшоты для отката перед правкой файлов.'
      },
      memory: {
        memoryEnabled: 'Сохранять долговременные заметки для будущих сессий.',
        userProfileEnabled: 'Вести компактный профиль предпочтений пользователя.'
      },
      context: {
        engine: 'Стратегия управления длинными диалогами у предела контекста.'
      },
      compression: {
        enabled: 'Сжимать старый контекст при росте диалога.'
      },
      voice: {
        autoTts: 'Автоматически озвучивать ответы ассистента.'
      },
      tts: {
        xai: {
          voiceId: 'ID голоса xAI (например, eve) или пользовательский ID.',
          language: 'Код языка (например, en, pt-BR) или «auto» для автоопределения.',
          speed: 'Скорость воспроизведения: 0.7 — медленнее, 1.0 — норма, 1.5 — быстрее.',
          autoSpeechTags:
            'Позволить LLM вставлять выразительные аудиотеги ([laughing], [sighs]) в текст перед синтезом.',
          optimizeStreamingLatency: 'Баланс задержки и качества: 0 — лучшее качество, 2 — наименьшая задержка.',
          sampleRate: 'Частота дискретизации в Гц. Выше — лучше качество, больше файлы.',
          bitRate: 'Битрейт MP3 в бит/с. Действует только при кодеке mp3.'
        },
        neutts: {
          device: 'Локальное устройство инференса для NeuTTS.'
        }
      },
      stt: {
        enabled: 'Включить транскрипцию речи локально или через провайдера.',
        echoTranscripts: 'Отправлять сырую 🎙️ расшифровку голосовых сообщений обратно в чат.',
        elevenlabs: {
          languageCode: 'Необязательный код ISO-639-3. Пусто — ElevenLabs определит сам.'
        }
      },
      updates: {
        nonInteractiveLocalChanges:
          'При обновлении из приложения (без терминала): сохранять локальные правки (stash) или отбрасывать (discard). Обновление из терминала всегда спрашивает.'
      }
    }),
    about: {
      heading: 'Hermes Desktop',
      version: value => `Версия ${value}`,
      versionUnavailable: 'Версия недоступна',
      updates: 'Обновления',
      checkNow: 'Проверить сейчас',
      checking: 'Проверка…',
      seeWhatsNew: 'Что нового',
      updateNow: 'Обновить сейчас',
      releaseNotes: 'Примечания к выпуску',
      onLatest: 'У вас последняя версия.',
      installing: 'Идёт установка обновления.',
      cantUpdate: 'Эта сборка не может обновить сама себя из приложения.',
      cantReach: 'Не удалось связаться с сервером обновлений.',
      tapCheck: 'Нажмите «Проверить сейчас» для поиска обновлений.',
      updateReady: count =>
        `Обновление готово (включает ${countRu(count, {
          one: 'изменение',
          few: 'изменения',
          many: 'изменений'
        })}).`,
      lastChecked: age => `Последняя проверка ${age}`,
      justNowSuffix: ' · только что',
      automaticUpdates: 'Автоматические обновления',
      automaticUpdatesDesc: 'Hermes автоматически проверяет обновления в фоне и уведомляет, когда одно готово.',
      branchCommit: (branch, commit) => `Ветвь ${branch} · Коммит ${commit}`,
      never: 'никогда',
      justNow: 'только что',
      minAgo: count =>
        `${countRu(count, {
          one: 'минуту',
          few: 'минуты',
          many: 'минут'
        })} назад`,
      hoursAgo: count =>
        `${countRu(count, {
          one: 'час',
          few: 'часа',
          many: 'часов'
        })} назад`,
      daysAgo: count =>
        `${countRu(count, {
          one: 'день',
          few: 'дня',
          many: 'дней'
        })} назад`
    },
    config: {
      none: 'Нет',
      noneParen: '(нет)',
      builtinOnly: 'Только встроенные',
      notSet: 'Не задано',
      commaSeparated: 'значения через запятую',
      loading: 'Загрузка конфигурации Hermes...',
      emptyTitle: 'Настраивать нечего',
      emptyDesc: 'В этом разделе нет настраиваемых параметров.',
      failedLoad: 'Не удалось загрузить настройки',
      autosaveFailed: 'Автосохранение не удалось',
      imported: 'Конфигурация импортирована',
      invalidJson: 'Неверный JSON конфигурации',
      keepAwakeTitle: 'Не давать компьютеру спать',
      keepAwakeDesc: 'Запретить сон, чтобы длинные или ночные задачи не прерывались. Экран может гаснуть.'
    },
    quickEntry: {
      enabledTitle: 'Быстрый ввод',
      enabledDesc:
        'Открывайте небольшое поле ввода глобальным сочетанием клавиш и отправляйте промпт, не открывая Hermes.',
      shortcutTitle: 'Сочетание для быстрого ввода',
      shortcutDesc: 'Нужна хотя бы одна клавиша-модификатор, например CommandOrControl+Shift+Space.',
      active: 'Сочетание клавиш активно.',
      takenBy: 'Это сочетание уже использует другое приложение — выберите другое.',
      invalidShortcut: 'Недопустимое сочетание. Добавьте хотя бы одну клавишу-модификатор.'
    },
    credentials: {
      pasteKey: 'Вставить ключ',
      pasteLabelKey: label => `Вставить ключ ${label}`,
      optional: 'Необязательно',
      enterValueFirst: 'Сначала введите значение.',
      couldNotSave: 'Не удалось сохранить учётные данные.',
      remove: 'Удалить',
      getKey: 'Получить ключ',
      saving: 'Сохранение'
    },
    envActions: {
      actionsFor: label => `Действия для ${label}`,
      credentialActions: 'Действия с учётными данными',
      manageInKeys: 'Управление в «API-ключи»',
      docs: 'Документация',
      hideValue: 'Скрыть значение',
      revealValue: 'Показать значение',
      replace: 'Заменить',
      set: 'Установить',
      clear: 'Очистить'
    },
    gateway: {
      loading: 'Загрузка настроек шлюза...',
      unavailableTitle: 'Настройки шлюза недоступны',
      unavailableDesc: 'IPC-мост десктопа не предоставляет настройки шлюза.',
      title: 'Подключение к шлюзу',
      envOverride: 'переопределение переменных окружения',
      intro:
        'Hermes Desktop по умолчанию запускает собственный локальный шлюз. Используйте удалённый шлюз, когда хотите, чтобы приложение управляло уже запущенным бэкендом Hermes на другой машине или за доверенным прокси. Выберите профиль ниже, чтобы задать ему собственный удалённый хост.',
      appliesTo: 'Применяется к',
      allProfiles: 'Все профили',
      defaultConnection: 'Подключение по умолчанию для каждого профиля без собственного переопределения.',
      profileConnection: profile =>
        `Подключение используется только когда «${profile}» — активный профиль. Выберите «Использовать шлюз по умолчанию», чтобы удалить переопределение.`,
      envOverrideTitle: 'Этой сессией десктопа управляют переменные окружения.',
      envOverrideDesc:
        'Сбросьте HERMES_DESKTOP_REMOTE_URL и HERMES_DESKTOP_REMOTE_TOKEN, чтобы использовать сохранённые ниже настройки.',
      modeTitle: 'Режим подключения',
      localTitle: 'Локальный шлюз',
      localDesc: 'Запустить частный бэкенд Hermes на localhost. Это значение по умолчанию, работает офлайн.',
      inheritTitle: 'Использовать шлюз по умолчанию',
      inheritDesc: 'Удалить переопределение профиля и использовать подключение по умолчанию.',
      remoteTitle: 'Удалённый шлюз',
      remoteDesc: 'Подключить этот десктоп к удалённому бэкенду Hermes.',
      remoteAuthHint:
        'Облачные шлюзы используют OAuth или логин и пароль; самостоятельно размещённые шлюзы могут использовать сессионный токен.',
      cloudTitle: 'Hermes Cloud',
      cloudDesc: 'Войдите в Hermes Cloud и выберите агентов на вашем аккаунте — без вставки URL.',
      cloudSignInTitle: 'Hermes Cloud',
      cloudSignIn: 'Войти в Hermes Cloud',
      cloudSignedIn: 'Вход в Hermes Cloud выполнен',
      cloudNeedsSignIn: 'Войдите в Hermes Cloud, чтобы обнаружить агентов на вашем аккаунте.',
      cloudSignedInDesc: 'Вы вошли. Выберите агента ниже; сессия обновится автоматически.',
      cloudAgentsTitle: 'Ваши агенты',
      cloudOrgPickerTitle: 'Выберите организацию',
      cloudOrgSelect: 'Выбрать',
      cloudOrgChange: 'Сменить организацию',
      cloudOrgRole: role => `Роль: ${role}`,
      cloudLoadingAgents: 'Загрузка ваших агентов…',
      cloudNoAgents: {
        before: 'Агенты на этом аккаунте не найдены. Создайте агента на ',
        linkText: 'портале Nous',
        after: ', затем обновите.'
      },
      cloudRefresh: 'Обновить',
      cloudConnect: 'Подключить',
      cloudConnecting: 'Подключение…',
      cloudDiscoverFailed: 'Не удалось загрузить агентов Hermes Cloud',
      cloudConnectFailed: 'Не удалось подключиться к этому агенту',
      cloudSignInFailed: 'Вход в Hermes Cloud не удался',
      cloudSignedOutTitle: 'Вы вышли из Hermes Cloud',
      cloudSignedOutMessage: 'Сессия Hermes Cloud очищена.',
      cloudConnectedTitle: 'Подключено',
      cloudConnectedPill: 'Подключено',
      cloudConnectedTo: name => `Подключено к ${name}.`,
      cloudAgentProvisioning: 'Подготовка…',
      cloudStatusLabel: status => `Статус: ${status}`,
      remoteUrlTitle: 'Удалённый URL',
      remoteUrlDesc: 'Базовый URL удалённого бэкенда. Поддерживаются префиксы путей, например /hermes.',
      probing: 'Проверка метода аутентификации шлюза…',
      probeError:
        'Пока не удалось связаться с этим шлюзом. Проверьте URL — метод аутентификации появится после ответа.',
      signedIn: 'Вход выполнен',
      signIn: 'Войти',
      signOut: 'Выйти',
      signInWith: provider => `Войти через ${provider}`,
      authTitle: 'Аутентификация',
      authSignedInPassword: 'Этот шлюз использует логин и пароль. Вы вошли; сессия обновляется автоматически.',
      authSignedInOauth: 'Этот шлюз использует OAuth. Вы вошли; сессия обновляется автоматически.',
      authNeedsPassword: 'Этот шлюз использует логин и пароль. Войдите, чтобы авторизовать это приложение.',
      authNeedsOauth: provider =>
        `Этот шлюз использует OAuth. Войдите через ${provider}, чтобы авторизовать это приложение.`,
      tokenTitle: 'Сессионный токен',
      tokenDesc: 'Токен сессии для REST и WebSocket. Оставьте пустым, чтобы сохранить существующий.',
      existingToken: value => `Существующий токен ${value}`,
      savedToken: 'сохранён',
      pasteSessionToken: 'Вставить сессионный токен',
      testRemote: 'Проверить удалённый шлюз',
      saveForRestart: 'Сохранить для следующего перезапуска',
      saveAndReconnect: 'Сохранить и переподключиться',
      diagnostics: 'Диагностика',
      diagnosticsDesc: 'Показать desktop.log в файловом менеджере — полезно при сбое запуска шлюза.',
      openLogs: 'Открыть логи',
      incompleteTitle: 'Удалённый шлюз настроен не полностью',
      incompleteSignIn: 'Введите удалённый URL и войдите перед переключением на удалённый режим.',
      incompleteToken: 'Введите удалённый URL и сессионный токен перед переключением на удалённый режим.',
      incompleteSignInTest: 'Введите удалённый URL и войдите перед тестом.',
      incompleteTokenTest: 'Введите удалённый URL и сессионный токен перед тестом.',
      enterUrlFirst: 'Сначала введите удалённый URL.',
      restartingTitle: 'Перезапуск подключения к шлюзу',
      savedTitle: 'Настройки шлюза сохранены',
      restartingMessage: 'Hermes Desktop переподключится с сохранёнными настройками — оболочка останется открытой.',
      savedMessage: 'Сохранено для следующего перезапуска.',
      connectedTo: (baseUrl, version) => `Подключено к ${baseUrl}${version ? ` · Hermes ${version}` : ''}`,
      reachableTitle: 'Удалённый шлюз доступен',
      signedOutTitle: 'Вы вышли',
      signedOutMessage: 'Сессия удалённого шлюза очищена.',
      failedLoad: 'Не удалось загрузить настройки шлюза',
      signInFailed: 'Вход не удался',
      signOutFailed: 'Выход не удался',
      testFailed: 'Тест удалённого шлюза не удался',
      applyFailed: 'Не удалось применить настройки шлюза',
      saveFailed: 'Не удалось сохранить настройки шлюза',
      sshTitle: 'Подключение по SSH',
      sshDesc:
        'Hermes запускается на удалённом хосте по SSH и туннелируется в это приложение — ничего не нужно запускать или открывать самому. Требуется рабочий SSH-доступ по ключу.',
      sshTrustHint: 'Первый представленный ключ хоста принимается и закрепляется; последующие изменения отклоняются.',
      sshHostTitle: 'Хост',
      sshHostDesc: 'user@host или алиас Host из ~/.ssh/config.',
      sshHostPick: 'Выберите хост…',
      sshHostPickTitle: 'Хост',
      sshHostPickDesc: 'Алиас Host из ~/.ssh/config или «Свой» для ручного ввода.',
      sshHostCustom: 'Свой (ввести вручную)…',
      sshUserTitle: 'Пользователь',
      sshUserDesc: 'Пусто — из ~/.ssh/config или текущий пользователь.',
      sshUserPlaceholder: 'из ~/.ssh/config',
      sshPortTitle: 'Порт',
      sshPortDesc: 'Пусто — 22 или порт из ~/.ssh/config.',
      sshKeyTitle: 'Файл ключа (IdentityFile)',
      sshKeyDesc: 'Путь к приватному ключу. Пусто — ssh-agent или ~/.ssh/config.',
      sshHermesPathTitle: 'Путь к Hermes (необязательно)',
      sshHermesPathDesc: 'Полный путь к бинарю hermes на хосте. Пусто — автоопределение.',
      sshHermesPathPlaceholder: 'автоопределение',
      sshTestConnection: 'Проверить SSH',
      sshConnect: 'Подключить',
      sshButtonsHint: '«Сохранить» применится при следующем запуске. «Подключить» переподключает сразу.',
      sshReachable: (host, platform) => `Доступен: ${host} (${platform}) — Hermes найден`,
      sshIncompleteHost: 'Укажите SSH-хост перед подключением.',
      sshErrUnreachable: 'Не удалось достучаться до хоста по SSH. Проверьте хост, порт и сеть.',
      sshErrAuth:
        'Ошибка аутентификации SSH. Загрузите ключ в ssh-agent (ssh-add) или задайте IdentityFile в ~/.ssh/config — Hermes запускает ssh неинтерактивно.',
      sshErrHostKey:
        'Ключ хоста ИЗМЕНИЛСЯ с прошлого подключения. Убедитесь, что это ожидаемо, затем выполните ssh-keygen -R <host> и переподключитесь.',
      sshErrNotInstalled:
        'Hermes не установлен на удалённом хосте. Установите его там (curl -fsSL https://hermes-agent.nousresearch.com/install.sh | sh) или укажите путь к Hermes.',
      sshErrPlatform: 'Неподдерживаемая платформа хоста. SSH-режим Hermes Desktop поддерживает Linux, macOS и Windows.',
      sshErrTimeout: 'Таймаут SSH-подключения. Хост недоступен или спит.',
      sshErrUpdateRequired: 'Обновите Hermes на удалённом хосте перед подключением через Desktop SSH.',
      sshErrUnknown: 'SSH-подключение не удалось.'
    },
    keys: {
      loading: 'Загрузка API-ключей и учётных данных...',
      failedLoad: 'Не удалось загрузить API-ключи',
      empty: 'В этой категории пока ничего не настроено.'
    },
    mcp: {
      loading: 'Загрузка MCP-серверов...',
      failedLoad: 'Не удалось загрузить конфигурацию MCP',
      nameRequiredTitle: 'Требуется имя',
      nameRequiredMessage: 'Задайте этому MCP-серверу конфигурационный ключ.',
      objectRequired: 'Конфигурация сервера должна быть JSON-объектом',
      invalidJson: 'Неверный MCP JSON',
      saveFailed: 'Не удалось сохранить',
      removeFailed: 'Не удалось удалить',
      gatewayUnavailableTitle: 'Шлюз недоступен',
      gatewayUnavailableMessage: 'Переподключите шлюз перед перезагрузкой MCP.',
      reloadedTitle: 'MCP-инструменты перезагружены',
      reloadedMessage: 'Новые схемы инструментов применяются к новым шагам.',
      reloadFailed: 'Не удалось перезагрузить MCP',
      savedTitle: 'MCP-сервер сохранён',
      savedMessage: name => `${name} применится после перезагрузки MCP.`,
      newServer: 'Новый сервер',
      reload: 'Перезагрузить MCP',
      reloading: 'Перезагрузка...',
      emptyTitle: 'Нет MCP-серверов',
      emptyDesc: 'Добавьте stdio или HTTP-сервер для предоставления MCP-инструментов.',
      disabled: 'отключён',
      editServer: 'Изменить сервер',
      name: 'Имя',
      serverJson: 'JSON сервера',
      remove: 'Удалить',
      saveServer: 'Сохранить сервер',
      test: 'Тест подключения',
      testing: 'Тестирование…',
      testOk: count =>
        `Подключено — доступно ${countRu(count, {
          one: 'инструмент',
          few: 'инструмента',
          many: 'инструментов'
        })}`,
      testFailed: 'Подключение не удалось',
      enableServer: name => `Включить ${name}`,
      disableServer: name => `Отключить ${name}`,
      serverEnabled: name => `${name} включён — применяется к новым сессиям.`,
      serverDisabled: name => `${name} отключён — применяется к новым сессиям.`,
      toggleFailed: name => `Не удалось переключить ${name}`,
      tabServers: 'Серверы',
      tabCatalog: 'Каталог',
      catalogLoading: 'Загрузка каталога MCP…',
      catalogLoadFailed: 'Не удалось загрузить каталог MCP',
      catalogEmpty: 'Нет записей в каталоге.',
      catalogInstalled: 'Установлен',
      catalogEnabled: 'Включён',
      catalogNeedsInstall: 'Требует сборки',
      catalogInstall: 'Установить',
      catalogInstalling: 'Установка…',
      catalogInstallStarted: name => `Установка ${name}… применится к новым сессиям после завершения.`,
      catalogInstallFailed: name => `Не удалось установить ${name}`,
      catalogEnvPrompt: name => `${name} требует учётные данные`,
      catalogEnvRequired: 'Заполните обязательные поля перед установкой.',
      capabilitySummary: (tools, prompts, resources) =>
        `Доступно: ${[
          countRu(tools, {
            one: 'инструмент',
            few: 'инструмента',
            many: 'инструментов'
          }),
          ...(prompts
            ? [
                countRu(prompts, {
                  one: 'промпт',
                  few: 'промпта',
                  many: 'промптов'
                })
              ]
            : []),
          ...(resources
            ? [
                countRu(resources, {
                  one: 'ресурс',
                  few: 'ресурса',
                  many: 'ресурсов'
                })
              ]
            : [])
        ].join(', ')}`,
      statusConnecting: 'Подключение…',
      statusNeedsAuth: 'Требуется аутентификация',
      statusError: 'Ошибка',
      statusOff: 'Выкл',
      allServers: 'Все серверы',
      authenticatedTitle: 'Аутентифицировано',
      authenticatedMessage: (server, count) =>
        `${server}: ${countRu(count, {
          one: 'инструмент',
          few: 'инструмента',
          many: 'инструментов'
        })}`,
      waitingForBrowser: 'Ожидание браузера…',
      authenticate: 'Аутентифицировать',
      unsavedConnect: 'Не сохранено — сохраните mcp.json для подключения.',
      enableTool: tool => `Включить ${tool}`,
      disableTool: tool => `Отключить ${tool}`,
      noOutput: 'Пока нет вывода.'
    },
    model: {
      loading: 'Загрузка конфигурации модели...',
      appliesDesc: 'Применяется к новым сессиям. Чтобы сменить модель в текущем чате, используйте меню у поля ввода.',
      provider: 'Провайдер',
      model: 'Модель',
      applying: 'Применение...',
      defaultsLabel: 'По умолчанию',
      reasoning: 'Рассуждения',
      reasoningOff: 'Выкл',
      defaultsFailed: 'Не удалось сохранить умолчания модели',
      auxiliaryTitle: 'Вспомогательные модели',
      resetAllToMain: 'Сбросить все на основную',
      auxiliaryDesc:
        'Вспомогательные задачи по умолчанию запускаются на основной модели. Назначьте выделенную модель любой задаче для переопределения.',
      setToMain: 'На основную',
      change: 'Изменить',
      autoUseMain: 'авто · основная модель',
      providerDefault: '(по умолчанию провайдера)',
      fallbackAdd: 'Добавить резервную модель',
      fallbackEmpty: 'Нет резервных моделей — используется модель по умолчанию, пока она не завершится с ошибкой.',
      notInCatalog:
        'отсутствует в списке моделей этого провайдера — запросы могут быть перенаправлены на резервную модель.',
      tasks: {
        vision: {
          label: 'Зрение',
          hint: 'Анализ изображений'
        },
        web_extract: {
          label: 'Извлечение веб',
          hint: 'Суммаризация страниц'
        },
        compression: {
          label: 'Сжатие',
          hint: 'Уплотнение контекста'
        },
        skills_hub: {
          label: 'Хаб навыков',
          hint: 'Поиск навыков'
        },
        approval: {
          label: 'Подтверждение',
          hint: 'Умное авто-одобрение'
        },
        mcp: {
          label: 'MCP',
          hint: 'Маршрутизация MCP-инструментов'
        },
        title_generation: {
          label: 'Ген. заголовков',
          hint: 'Заголовки сессий'
        },
        curator: {
          label: 'Куратор',
          hint: 'Ревью использования навыков'
        }
      }
    },
    providers: {
      connectAccount: 'Подключить аккаунт',
      haveApiKey: 'У вас есть API-ключ?',
      intro:
        'Войдите через подписку — не нужно копировать API-ключ. Hermes запускает вход через браузер прямо в приложении.',
      connected: 'Подключено',
      collapse: 'Свернуть',
      connectAnother: 'Подключить другого провайдера',
      otherProviders: 'Другие провайдеры',
      disconnect: 'Отключить',
      disconnectInTerminal: 'Отключить (запускает команду удаления в терминале)',
      removeConfirm: provider => `Удалить ${provider}?`,
      removeExternalGeneric: provider => `${provider} управляется собственным CLI — удалите там.`,
      removeKeyManaged: provider => `${provider} настроен через API-ключ. Удалите его из раздела «API-ключи».`,
      removeTerminalConfirm: (provider, command) =>
        `Отключить ${provider}? Это запустит «${command}» в терминале для очистки учётных данных.`,
      removeTerminalRunning: provider => `Отключение ${provider} в терминале…`,
      removedTitle: 'Аккаунт удалён',
      removedMessage: provider => `${provider} был удалён.`,
      failedRemove: provider => `Не удалось удалить ${provider}`,
      noProviderKeys: 'Нет доступных API-ключей провайдеров.',
      searchKeys: 'Поиск провайдеров…',
      noKeysMatch: 'Нет провайдеров, соответствующих вашему запросу.',
      localEndpoint: {
        title: 'Локальный или собственный эндпоинт',
        description: 'Укажите любой OpenAI-совместимый эндпоинт (Zyphra, vLLM, llama.cpp, Ollama и т. д.).'
      },
      loading: 'Загрузка провайдеров...'
    },
    sessions: {
      loading: 'Загрузка архивных сессий…',
      archivedTitle: 'Архивные сессии',
      archivedIntro:
        'Архивные чаты скрыты из боковой панели, но сохраняют все сообщения. Ctrl/⌘-клик по чату в боковой панели архивирует его.',
      emptyArchivedTitle: 'Архив пуст',
      emptyArchivedDesc: 'Архивируйте чат, чтобы он появился здесь.',
      unarchive: 'Вернуть из архива',
      deletePermanently: 'Удалить навсегда',
      messages: count =>
        countRu(count, {
          one: 'сообщение',
          few: 'сообщения',
          many: 'сообщений'
        }),
      restored: 'Восстановлено',
      deleteConfirm: title => `Навсегда удалить «${title}»? Это действие необратимо.`,
      autoArchiveTitle: 'Автоархивация старых чатов',
      autoArchiveDesc:
        'Автоматически архивировать чаты, к которым вы давно не возвращались. Закреплённые не архивируются, ничего не удаляется — архивированные просто переносятся сюда.',
      autoArchiveDaysLabel: 'Архивировать через',
      autoArchiveDaysUnit: 'дней неактивности',
      autoArchiveFailed: 'Не удалось обновить автоархивацию',
      defaultDirTitle: 'Каталог проекта по умолчанию',
      defaultDirDesc:
        'Новые сессии начинаются в этой папке, если не выбрана другая. Оставьте пустым для использования домашнего каталога.',
      defaultDirUpdated: 'Каталог проекта по умолчанию обновлён — начните новый чат (Ctrl/⌘+N) для применения',
      defaultsTo: label => `По умолчанию ${label}.`,
      change: 'Изменить',
      choose: 'Выбрать',
      clear: 'Очистить',
      notSet: 'Не задано',
      failedLoad: 'Не удалось загрузить архивные сессии',
      unarchiveFailed: 'Не удалось вернуть из архива',
      deleteFailed: 'Не удалось удалить',
      updateDirFailed: 'Не удалось обновить каталог по умолчанию',
      clearDirFailed: 'Не удалось очистить каталог по умолчанию'
    },
    toolsets: {
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
      selectedMessage: provider => `${provider} теперь активен.`,
      failedSelect: provider => `Не удалось выбрать ${provider}`,
      failedLoad: 'Не удалось загрузить конфигурацию инструментов',
      noProviderOptions:
        'У этого набора инструментов нет выбора провайдера — включите и он работает с текущей конфигурацией.',
      noProviders: 'Для этого набора инструментов сейчас нет доступных провайдеров.',
      ready: 'Готово',
      needsSignIn: 'Требуется вход',
      needsSetup: 'Требуется настройка',
      nousIncluded: 'Входит в подписку Nous — войдите в Nous Portal для активации.',
      nousAuthNeededTitle: 'Войдите в Nous Portal',
      nousAuthNeededMessage: provider => `${provider} сохранён, но не активируется, пока вы не войдёте в Nous Portal.`,
      nousAuthSignIn: 'Войти',
      nousAuthDoneTitle: 'Nous Portal подключён',
      nousAuthDoneMessage: 'Ваши бэкенды по подписке активны.',
      nousAuthFailed: 'Вход в Nous Portal не завершился',
      noApiKeyRequired: 'API-ключ не требуется.',
      postSetupHint: step =>
        `Этот бэкенд требует одноразовой установки (${step}). Запускается на этой машине — может занять несколько минут.`,
      postSetupInstalledHint: 'Установлено. Повторная настройка нужна только при поломке.',
      postSetupRun: 'Запустить настройку',
      postSetupRerun: 'Повторить настройку',
      postSetupInstalled: 'Установлено',
      postSetupRunning: 'Установка…',
      postSetupStarting: 'Запуск…',
      postSetupCompleteTitle: 'Настройка завершена',
      postSetupCompleteMessage: step => `${step} установлен.`,
      postSetupErrorTitle: 'Настройка завершена с ошибками',
      postSetupErrorMessage: step => `Проверьте лог ${step}.`,
      postSetupFailed: step => `Не удалось запустить настройку ${step}`,
      webSearchActive: backend => `Поиск: ${backend}`,
      webExtractActive: backend => `Извлечение: ${backend}`,
      webCapabilityUnset: 'не задано',
      webUseForSearch: 'Использовать для поиска',
      webUseForExtract: 'Использовать для извлечения',
      webUsedForSearch: 'Бэкенд поиска',
      webUsedForExtract: 'Бэкенд извлечения',
      webCapabilitySelectedMessage: (provider, capability) =>
        `${provider} теперь отвечает за веб-${capability === 'search' ? 'поиск' : 'извлечение'}.`,
      failedSelectCapability: provider => `Не удалось выбрать ${provider}`,
      loadingModels: 'Загрузка каталога моделей…',
      modelSectionTitle: 'Модель',
      modelCount: count =>
        countRu(count, {
          one: 'модель',
          few: 'модели',
          many: 'моделей'
        }),
      modelInUse: 'Используется',
      modelDefault: 'по умолчанию',
      modelInactiveHint: 'Сначала выберите этот бэкенд для смены его модели.',
      modelSelectedTitle: 'Модель выбрана',
      modelSelectedMessage: model => `${model} применяется к новым сессиям.`,
      failedSelectModel: model => `Не удалось выбрать ${model}`,
      terminalBackend: {
        sectionTitle: 'Бэкенд выполнения',
        loading: 'Проверка бэкендов выполнения…',
        failedLoad: 'Не удалось загрузить бэкенды терминала',
        ready: 'Готов',
        needsSetup: 'Требуется настройка',
        unavailable: 'Недоступен',
        inUse: 'Используется',
        selectedTitle: 'Бэкенд выбран',
        selectedMessage: backend =>
          `Команды терминала теперь выполняются через ${backend}. Применяется к новым сессиям.`,
        failedSelect: backend => `Не удалось выбрать ${backend}`,
        needsSetupHint: 'Бэкенд можно выбрать сейчас — команды будут падать до завершения настройки.'
      }
    }
  },
  skills: {
    tabSkills: 'Навыки',
    tabToolsets: 'Наборы инструментов',
    tabMcp: 'MCP',
    tabHub: 'Обзор хаба',
    all: 'Все',
    searchSkills: 'Поиск навыков...',
    searchToolsets: 'Поиск наборов инструментов...',
    refresh: 'Обновить навыки',
    refreshing: 'Обновление навыков',
    loading: 'Загрузка…',
    noSkillsTitle: 'Навыки не найдены',
    noSkillsDesc: 'Попробуйте более широкий поиск или другую категорию.',
    noToolsetsTitle: 'Наборы инструментов не найдены',
    noToolsetsDesc: 'Попробуйте более широкий поисковый запрос.',
    noDescription: 'Нет описания.',
    configured: 'Настроено',
    needsKeys: 'Нужны ключи',
    visionModelHint:
      'Vision использует конфигурацию вспомогательной модели — модель с поддержкой изображений выбирается там, а не по провайдерам здесь.',
    visionModelLink: 'Выбрать vision-модель в Настройки → Модели',
    toolsetsEnabled: (enabled, total) =>
      `Включено ${enabled}/${total} ${pluralRu(total, {
        one: 'набор инструментов',
        few: 'набора инструментов',
        many: 'наборов инструментов'
      })}`,
    configureToolset: label => `Настроить ${label}`,
    toggleToolset: label => `Переключить набор инструментов «${label}»`,
    skillsLoadFailed: 'Не удалось загрузить навыки',
    toolsetsRefreshFailed: 'Не удалось обновить наборы инструментов',
    skillEnabled: 'Навык включён',
    skillDisabled: 'Навык отключён',
    toolsetEnabled: 'Набор инструментов включён',
    toolsetDisabled: 'Набор инструментов отключён',
    appliesToNewSessions: name => `${name} применяется к новым сессиям.`,
    failedToUpdate: name => `Не удалось обновить ${name}`,
    sortMostUsed: 'Часто используемые',
    sortAlpha: 'А–Я',
    sortMostUsedDesc: '↓ Часто используемые',
    sortLeastUsedAsc: '↑ Редко используемые',
    enableAll: 'Включить все',
    disableAll: 'Отключить все',
    disableUnused: 'Отключить неиспользуемые',
    bulkUpdated: count =>
      `Обновлено ${countRu(count, {
        one: 'элемент',
        few: 'элемента',
        many: 'элементов'
      })} для новых сессий.`,
    bulkNoChange: 'Нечего менять.',
    usageCount: count => `использован ${count}×`,
    provenance: {
      agent: 'Изучен',
      bundled: 'Встроен',
      hub: 'Хаб'
    },
    emptyNoneFound: noun => `${noun} не найдены`,
    emptyNothingMatches: query => `Ничего не соответствует «${query}».`,
    emptyNoneAvailable: noun => `Нет доступных ${noun}.`,
    changesApplyNewSessions: 'Изменения применяются к новым сессиям.',
    skillUpdated: 'Навык обновлён',
    edit: 'Редактировать',
    archive: 'Архивировать',
    skillArchivedTitle: 'Навык архивирован',
    skillArchivedMessage: 'Восстанавливается через hermes curator restore.',
    hub: {
      searchPlaceholder: 'Поиск по хабу навыков',
      search: 'Поиск',
      searching: 'Поиск...',
      connectingHubs: 'Подключение к хабам навыков...',
      connectedHubs: 'Подключённые источники:',
      featured: 'Избранные навыки',
      landingHint: 'Ищите в хабе для просмотра устанавливаемых навыков из официального индекса, GitHub и сообщества.',
      noResults: 'В хабе не найдено подходящих навыков.',
      resultCount: (count, ms) =>
        `${countRu(count, {
          one: 'результат',
          few: 'результата',
          many: 'результатов'
        })}${ms !== null ? ` за ${ms} мс` : ''}`,
      timedOut: sources => `Таймаут: ${sources}`,
      installed: 'Установлен',
      install: 'Установить',
      installing: 'Установка...',
      uninstall: 'Удалить',
      uninstalling: 'Удаление...',
      updateAll: 'Обновить установленные',
      updating: 'Обновление...',
      preview: 'Предпросмотр',
      scan: 'Сканировать',
      scanning: 'Сканирование...',
      close: 'Закрыть',
      files: 'Файлы',
      noReadme: 'У этого навыка нет предпросмотра SKILL.md.',
      trust: {
        builtin: 'встроен',
        trusted: 'доверенный',
        community: 'сообщество'
      },
      verdictSafe: 'Безопасно',
      verdictCaution: 'Осторожно',
      verdictDangerous: 'Опасно',
      policyAllow: 'Установка разрешена',
      policyAsk: 'Проверка перед установкой',
      policyBlock: 'Установка заблокирована политикой',
      findings: count =>
        countRu(count, {
          one: 'находка',
          few: 'находки',
          many: 'находок'
        }),
      noFindings: 'Уязвимостей не найдено.',
      installStarted: name => `Установка ${name}...`,
      uninstallStarted: name => `Удаление ${name}...`,
      updateStarted: 'Обновление установленных навыков...',
      actionFailed: 'Действие с навыком не удалось',
      actionLog: 'Лог действий',
      loadFailed: 'Не удалось загрузить хаб навыков',
      previewFailed: 'Не удалось показать предпросмотр навыка',
      scanFailed: 'Сканирование безопасности не удалось',
      searchFailed: 'Поиск по хабу не удался'
    }
  },
  starmap: {
    title: 'Граф памяти',
    subtitle: (nodes, clusters) =>
      `${countRu(nodes, {
        one: 'навык',
        few: 'навыка',
        many: 'навыков'
      })} в ${countRu(clusters, {
        one: 'категории',
        few: 'категориях',
        many: 'категориях'
      })}`,
    close: 'Закрыть граф памяти',
    refresh: 'Обновить',
    memory: 'Память',
    filterAll: 'Все',
    filterUsed: 'Использованные',
    filterLearned: 'Изученные',
    viewGraph: 'Граф',
    loadFailed: 'Не удалось загрузить граф памяти',
    loading: 'Загрузка…',
    emptyTitle: 'Пока ничего не изучено',
    emptyDesc: 'По мере того как Hermes создаёт навыки и память для вашей работы, они появятся здесь.',
    share: 'Поделиться картой',
    shareHint:
      'Скопируйте код, чтобы поделиться этой картой, или вставьте для загрузки. Включает только раскладку, не текст памяти или навыков.',
    shareTitle: 'Импорт / экспорт карты',
    sharePlaceholder: 'Вставьте код карты…',
    copy: 'Копировать код карты',
    copied: 'Скопировано!',
    importMap: 'Импорт карты',
    importBtn: 'Загрузить',
    importEmpty: 'Вставьте код карты для загрузки.',
    importSuccess: nodes =>
      `Загружена карта с ${countRu(nodes, {
        one: 'узлом',
        few: 'узлами',
        many: 'узлами'
      })}.`,
    importedBadge: 'импортированная карта',
    resetToMine: 'Вернуться к моей карте'
  },
  agents: {
    close: 'Закрыть агентов',
    title: 'Дерево порождений',
    subtitle: 'Активность субагентов для текущего шага.',
    emptyTitle: 'Нет активных субагентов',
    emptyDesc: 'Когда шаг делегирует работу, дочерние агенты стримят прогресс сюда.',
    running: 'Выполняется',
    failed: 'Ошибка',
    done: 'Готово',
    streaming: 'Стриминг',
    files: 'Файлы',
    moreFiles: count =>
      `+ ещё ${countRu(count, {
        one: 'файл',
        few: 'файла',
        many: 'файлов'
      })}`,
    delegation: index => `Делегирование ${index}`,
    workers: count =>
      countRu(count, {
        one: 'воркер',
        few: 'воркера',
        many: 'воркеров'
      }),
    workersActive: count =>
      countRu(count, {
        one: 'активен',
        few: 'активны',
        many: 'активны'
      }),
    agentsCount: count =>
      countRu(count, {
        one: 'агент',
        few: 'агента',
        many: 'агентов'
      }),
    activeCount: count =>
      countRu(count, {
        one: 'активен',
        few: 'активны',
        many: 'активны'
      }),
    failedCount: count =>
      countRu(count, {
        one: 'с ошибкой',
        few: 'с ошибками',
        many: 'с ошибками'
      }),
    toolsCount: count =>
      countRu(count, {
        one: 'инструмент',
        few: 'инструмента',
        many: 'инструментов'
      }),
    filesCount: count =>
      countRu(count, {
        one: 'файл',
        few: 'файла',
        many: 'файлов'
      }),
    updatedAgo: age => `обновлено ${age}`,
    ageNow: 'сейчас',
    ageSeconds: seconds => `${seconds}с назад`,
    ageMinutes: minutes => `${minutes}м назад`,
    ageHours: hours => `${hours}ч назад`,
    ageDays: days => `${days} дн. назад`,
    durationSeconds: seconds => `${seconds}с`,
    durationMinutes: (minutes, seconds) => `${minutes}м ${seconds}с`,
    tokens: value => `${value} ток`
  },
  commandCenter: {
    close: 'Закрыть командный центр',
    paletteTitle: 'Палитра команд',
    back: 'Назад',
    searchPlaceholder: 'Поиск сессий, представлений и действий',
    goTo: 'Перейти к',
    goToSession: 'Перейти к сессии',
    branches: 'Ветви',
    commands: 'Команды',
    startInBranch: branch => `Новый диалог в ${branch}`,
    commandCenter: 'Командный центр',
    appearance: 'Внешний вид',
    settings: 'Настройки',
    changeTheme: 'Сменить тему',
    changeColorMode: 'Сменить цветовой режим…',
    pets: {
      title: 'Питомцы',
      placeholder: 'Поиск питомцев…',
      loading: 'Загрузка галереи petdex…',
      error: 'Не удалось связаться с галереей petdex.',
      staleBackend: 'Перезапустите Hermes для использования питомцев — бэкенд старше этой функции.',
      empty: 'Нет подходящих питомцев.',
      turnOff: 'Выключить',
      turnOn: 'Включить',
      installed: 'Установлен',
      generatedTag: 'Сгенерирован',
      adoptFailed: 'Не удалось установить питомца.',
      toggleFailed: 'Не удалось переключить питомца.',
      noneAvailable: 'Нет доступных питомцев — выберите ниже для установки.'
    },
    generatePet: {
      title: 'Сгенерировать питомца',
      placeholder: 'Опишите питомца для генерации…',
      promptHint: 'Введите описание, затем нажмите Enter для создания четырёх образов.',
      readyHint: 'Нажмите Enter для создания четырёх образов из вашего описания.',
      generate: 'Сгенерировать',
      generating: 'Генерация…',
      retry: 'Повторить',
      hatch: 'Вылупить',
      spawning: 'Создание…',
      hatching: 'Вылупливание питомца…',
      hatchingSub: 'Оживление…',
      hatched: 'Вылупился!',
      hatchRow: (_state, done, total) => `Рисую кадр ${done} из ${total}…`,
      hatchComposing: 'Сборка…',
      hatchSaving: 'Почти готово…',
      namePlaceholder: 'Назовите питомца',
      staleBackend: 'Обновите Hermes для генерации питомцев.',
      backgroundHint: 'Можно закрыть — Hermes уведомит, когда будет готово.',
      slowProviderHint: 'Это может занять несколько минут',
      remix: 'Вариация',
      remixConfirmTitle: 'Создать вариацию этого образа?',
      remixConfirmBody:
        'Создаёт новый набор черновиков на основе этого как отправной точки. Может занять несколько минут.',
      genericError: 'Генерация не удалась — попробуйте снова или выберите предложение.',
      referenceImageTooLarge: 'Референсное изображение слишком большое. Используйте до 16 МБ.',
      referenceImageInvalid: 'Не удалось прочитать референсное изображение. Попробуйте PNG, JPG, WebP или GIF.',
      adopt: 'Усыновить',
      startOver: 'Начать заново'
    },
    installTheme: {
      title: 'Установить тему…',
      pageTitle: 'Установить тему',
      placeholder: 'Поиск в VS Code Marketplace...',
      loading: 'Поиск в Marketplace...',
      error: 'Не удалось связаться с Marketplace.',
      empty: 'Нет подходящих тем.',
      install: 'Установить',
      installing: 'Установка...',
      installed: 'Установлено',
      installs: count =>
        `${count} ${pluralRu(Number(count), {
          one: 'установка',
          few: 'установки',
          many: 'установок'
        })}`
    },
    settingsFields: 'Поля настроек',
    mcpServers: 'MCP-серверы',
    archivedChats: 'Архив чатов',
    sections: {
      maintenance: 'Обслуживание',
      sessions: 'Сессии',
      system: 'Система',
      usage: 'Использование'
    },
    sectionDescriptions: {
      maintenance: 'Диагностика, резервные копии, куратор и данные памяти',
      sessions: 'Поиск и управление сессиями',
      system: 'Статус, логи и системные действия',
      usage: 'Токены, стоимость и активность навыков по времени'
    },
    nav: {
      newChat: {
        title: 'Новая сессия',
        detail: 'Начать новую сессию'
      },
      settings: {
        title: 'Настройки',
        detail: 'Настроить десктоп Hermes'
      },
      skills: {
        title: 'Навыки и инструменты',
        detail: 'Навыки, инструменты и MCP-серверы'
      },
      messaging: {
        title: 'Мессенджеры',
        detail: 'Настроить Telegram, Slack, Discord и др.'
      },
      artifacts: {
        title: 'Артефакты',
        detail: 'Просмотр сгенерированных результатов'
      }
    },
    sectionEntries: {
      sessions: {
        title: 'Панель сессий',
        detail: 'Поиск, закрепление и управление сессиями'
      },
      system: {
        title: 'Системная панель',
        detail: 'Статус шлюза, логи, перезапуск/обновление'
      },
      usage: {
        title: 'Панель использования',
        detail: 'Токены, стоимость и активность навыков'
      }
    },
    providerNavigate: 'Перейти',
    providerSessions: 'Сессии',
    refresh: 'Обновить',
    refreshing: 'Обновление...',
    noResults: 'Совпадений не найдено.',
    pinSession: 'Закрепить сессию',
    unpinSession: 'Открепить сессию',
    exportSession: 'Экспорт сессии',
    deleteSession: 'Удалить сессию',
    noSessions: 'Сессий пока нет.',
    gatewayRunning: 'Шлюз мессенджеров запущен',
    gatewayStopped: 'Шлюз мессенджеров остановлен',
    hermesActiveSessions: (version, count) =>
      `Hermes ${version} · ${countRu(count, {
        one: 'активная сессия',
        few: 'активные сессии',
        many: 'активных сессий'
      })}`,
    restartGateway: 'Перезапустить шлюз',
    gatewayRestartFailed: 'Не удалось перезапустить шлюз.',
    updateHermes: 'Обновить Hermes',
    actionRunning: 'выполняется',
    actionDone: 'готово',
    actionFailed: 'ошибка',
    actionStartedWaiting: 'Действие запущено, ожидание статуса...',
    loadingStatus: 'Загрузка статуса...',
    recentLogs: 'Недавние логи',
    noLogs: 'Логи пока не загружены.',
    days: count => `${count}д`,
    statSessions: 'Сессии',
    statApiCalls: 'API-вызовы',
    statTokens: 'Токены вх/вых',
    statCost: 'Стоимость (оценка)',
    actualCost: cost => `фактически ${cost}`,
    loadingUsage: 'Загрузка статистики...',
    noUsage: period =>
      `Нет данных за ${countRu(period, {
        one: 'день',
        few: 'дня',
        many: 'дней'
      })}.`,
    retry: 'Повторить',
    dailyTokens: 'Токены по дням',
    input: 'вход',
    output: 'выход',
    noDailyActivity: 'Нет дневной активности.',
    topModels: 'Топ моделей',
    noModelUsage: 'Нет использования моделей.',
    topSkills: 'Топ навыков',
    noSkillActivity: 'Нет активности навыков.',
    actions: count =>
      `${count} ${pluralRu(Number(count), {
        one: 'действие',
        few: 'действия',
        many: 'действий'
      })}`,
    logFile: 'Файл лога',
    logLevel: 'Уровень',
    logSearchPlaceholder: 'Фильтр строк лога...',
    maintenance: {
      runOps: 'Диагностика',
      doctor: 'Запустить doctor',
      doctorDesc: 'Проверка установки, конфигурации и провайдеров',
      securityAudit: 'Аудит безопасности',
      securityAuditDesc: 'Сканирование конфигурации и навыков на рискованные настройки',
      backup: 'Создать резервную копию',
      backupDesc: 'Архивировать конфигурацию, память, навыки и сессии',
      debugShare: 'Отладочный отчёт',
      debugShareDesc: 'Загрузить цензурированный отчёт + логи, получить ссылки (авто-удаление через 6ч)',
      debugShareRunning: 'Загрузка отладочного отчёта...',
      debugShareLinks: 'Ссылки',
      debugShareFailed: 'Не удалось загрузить отладочный отчёт',
      copyLink: 'Копировать ссылку',
      linkCopied: 'Ссылка скопирована',
      curator: 'Куратор навыков',
      curatorDesc: 'Фоновая проверка, архивирующая устаревшие навыки агента',
      curatorPaused: 'Приостановлен',
      curatorActive: 'Активен',
      curatorDisabled: 'Отключён',
      curatorLastRun: when => `Последний запуск ${when}`,
      curatorNeverRan: 'Никогда не запускался',
      pause: 'Пауза',
      resume: 'Возобновить',
      runNow: 'Запустить сейчас',
      memoryData: 'Данные памяти',
      memoryDataDesc: 'Встроенные файлы памяти, внедряемые в каждую сессию',
      memoryProvider: name => `Активный провайдер: ${name}`,
      builtinMemory: 'встроенный',
      memoryFile: 'Память агента (MEMORY.md)',
      userFile: 'Профиль пользователя (USER.md)',
      bytes: size => size,
      empty: 'пусто',
      resetMemory: 'Сбросить память',
      resetUser: 'Сбросить профиль',
      resetAll: 'Сбросить оба',
      resetConfirm: target => `Удалить ${target}? Это действие необратимо.`,
      resetDone: files => `Удалено ${files}.`,
      resetFailed: 'Не удалось сбросить память',
      actionStarted: name => `${name} запущен — отслеживание лога...`,
      actionFailed: name => `${name} не удалось запустить`,
      running: 'Выполнение...',
      viewLog: 'Лог действий'
    }
  },
  messaging: {
    search: 'Поиск мессенджеров...',
    loading: 'Загрузка платформ...',
    loadFailed: 'Не удалось загрузить платформы мессенджеров',
    states: {
      connected: 'Подключено',
      connecting: 'Подключение',
      disabled: 'Отключено',
      fatal: 'Ошибка',
      gateway_stopped: 'Шлюз мессенджеров остановлен',
      not_configured: 'Требуется настройка',
      pending_restart: 'Требуется перезапуск',
      retrying: 'Повтор',
      startup_failed: 'Сбой запуска'
    },
    unknown: 'Неизвестно',
    hintPendingRestart: 'Перезапустите шлюз из статус-бара для применения этого изменения.',
    hintGatewayStopped: 'Запустите шлюз из статус-бара для подключения.',
    credentialsSet: 'Учётные данные заданы',
    needsSetup: 'Требуется настройка',
    gatewayStopped: 'Шлюз мессенджеров остановлен',
    getCredentials: 'Получить учётные данные',
    openSetupGuide: 'Открыть руководство по настройке',
    required: 'Обязательно',
    recommended: 'Рекомендуется',
    advanced: count => `Дополнительно (${count})`,
    noTokenNeeded: 'Этой платформе не нужен токен здесь. Используйте руководство выше, затем включите её ниже.',
    enabled: 'Включено',
    disabled: 'Отключено',
    unsavedChanges: 'Несохранённые изменения',
    saving: 'Сохранение...',
    saveChanges: 'Сохранить изменения',
    saved: 'Сохранено',
    replaceValue: 'Заменить текущее значение',
    openDocs: 'Открыть документацию',
    clearField: key => `Очистить ${key}`,
    enableAria: name => `Включить ${name}`,
    disableAria: name => `Отключить ${name}`,
    platformEnabled: name => `${name} включён`,
    platformDisabled: name => `${name} отключён`,
    restartToApply: 'Это изменение вступит в силу после перезапуска шлюза.',
    setupSaved: name => `Настройка ${name} сохранена`,
    restartToReconnect: 'Новые учётные данные вступят в силу после перезапуска шлюза.',
    keyCleared: key => `${key} очищен`,
    setupUpdated: name => `Настройка ${name} обновлена.`,
    failedUpdate: name => `Не удалось обновить ${name}`,
    failedSave: name => `Не удалось сохранить ${name}`,
    failedClear: key => `Не удалось очистить ${key}`,
    fieldCopy: {
      TELEGRAM_BOT_TOKEN: {
        label: 'Токен бота',
        help: 'Создайте бота через @BotFather, затем вставьте полученный токен.',
        placeholder: 'Вставьте токен Telegram-бота'
      },
      TELEGRAM_ALLOWED_USERS: {
        label: 'Разрешённые ID пользователей Telegram',
        help: 'Рекомендуется. Числовые ID через запятую от @userinfobot. Без этого кто угодно может написать вашему боту.'
      },
      TELEGRAM_PROXY: {
        label: 'URL прокси',
        help: 'Нужен только в сетях, где Telegram заблокирован.'
      },
      DISCORD_BOT_TOKEN: {
        label: 'Токен бота',
        help: 'Создайте приложение в Discord Developer Portal, добавьте бота, затем вставьте его токен.'
      },
      DISCORD_ALLOWED_USERS: {
        label: 'Разрешённые ID пользователей Discord',
        help: 'Рекомендуется. ID пользователей Discord через запятую.'
      },
      DISCORD_REPLY_TO_MODE: {
        label: 'Стиль ответов',
        help: 'first, all или off.'
      },
      DISCORD_ALLOW_ALL_USERS: {
        label: 'Разрешить всем пользователям Discord',
        help: 'Только для разработки. Если true, кто угодно может написать боту без списка разрешений.'
      },
      DISCORD_HOME_CHANNEL: {
        label: 'ID основного канала',
        help: 'Канал для проактивных сообщений бота (вывод cron, напоминания).'
      },
      DISCORD_HOME_CHANNEL_NAME: {
        label: 'Имя основного канала',
        help: 'Отображаемое имя основного канала в логах и статусе.'
      },
      BLUEBUBBLES_ALLOW_ALL_USERS: {
        label: 'Разрешить всех пользователей iMessage',
        help: 'Если true, пропустить список разрешений BlueBubbles.'
      },
      MATTERMOST_ALLOW_ALL_USERS: {
        label: 'Разрешить всех пользователей Mattermost'
      },
      MATTERMOST_HOME_CHANNEL: {
        label: 'Основной канал'
      },
      QQ_ALLOW_ALL_USERS: {
        label: 'Разрешить всех пользователей QQ'
      },
      QQBOT_HOME_CHANNEL: {
        label: 'Основной канал QQ',
        help: 'Канал/группа по умолчанию для доставки cron.'
      },
      QQBOT_HOME_CHANNEL_NAME: {
        label: 'Имя основного канала QQ'
      },
      SLACK_BOT_TOKEN: {
        label: 'Токен Slack-бота',
        help: 'Используйте токен бота из OAuth & Permissions после установки приложения Slack.',
        placeholder: 'Вставьте токен Slack-бота'
      },
      SLACK_APP_TOKEN: {
        label: 'Токен Slack-приложения',
        help: 'Используйте токен уровня приложения для Socket Mode.',
        placeholder: 'Вставьте токен Slack-приложения'
      },
      SLACK_ALLOWED_USERS: {
        label: 'Разрешённые ID пользователей Slack',
        help: 'Рекомендуется. ID пользователей Slack через запятую.'
      },
      MATTERMOST_URL: {
        label: 'URL сервера',
        placeholder: 'https://mattermost.example.com'
      },
      MATTERMOST_TOKEN: {
        label: 'Токен бота'
      },
      MATTERMOST_ALLOWED_USERS: {
        label: 'Разрешённые ID пользователей',
        help: 'Рекомендуется. ID пользователей Mattermost через запятую.'
      },
      MATRIX_HOMESERVER: {
        label: 'URL домашнего сервера',
        placeholder: 'https://matrix.org'
      },
      MATRIX_ACCESS_TOKEN: {
        label: 'Токен доступа'
      },
      MATRIX_USER_ID: {
        label: 'ID пользователя-бота',
        placeholder: '@hermes:example.org'
      },
      MATRIX_ALLOWED_USERS: {
        label: 'Разрешённые ID пользователей Matrix',
        help: 'Рекомендуется. ID в формате @user:server через запятую.'
      },
      SIGNAL_HTTP_URL: {
        label: 'URL моста Signal',
        placeholder: 'http://127.0.0.1:8080',
        help: 'URL запущенного REST-моста signal-cli.'
      },
      SIGNAL_ACCOUNT: {
        label: 'Номер телефона',
        help: 'Номер, зарегистрированный в мосте signal-cli.'
      },
      SIGNAL_ALLOWED_USERS: {
        label: 'Разрешённые пользователи Signal',
        help: 'Рекомендуется. Идентификаторы Signal через запятую.'
      },
      WHATSAPP_ENABLED: {
        label: 'Включить мост WhatsApp',
        help: 'Устанавливается переключателем ниже. Не меняйте, если не уверены.'
      },
      WHATSAPP_MODE: {
        label: 'Режим моста'
      },
      WHATSAPP_ALLOWED_USERS: {
        label: 'Разрешённые пользователи WhatsApp',
        help: 'Рекомендуется. Номера телефонов или WhatsApp ID через запятую.'
      }
    },
    platformIntro: {}
  },
  webhooks: {
    search: 'Поиск вебхуков…',
    loading: 'Загрузка вебхуков…',
    loadFailed: 'Не удалось загрузить вебхуки',
    subscriptions: (count: number) =>
      `${pluralRu(count, {
        one: 'Подписка',
        few: 'Подписки',
        many: 'Подписки'
      })} (${count})`,
    hint: 'Изменения подписок применяются на лету, когда приёмник запущен. Отключённые подписки отклоняют входящие события.',
    empty: 'Подписок на вебхуки пока нет.',
    disabledTitle: 'Приёмник вебхуков отключён',
    disabledBody:
      'Вебхуки — отдельная платформа шлюза. Включите их здесь, чтобы принимать входящие HTTP-события; каналы чатов нужны, только когда подписка доставляет в Telegram, Discord, Slack или другой канал.',
    enable: 'Включить вебхуки',
    enabling: 'Включение…',
    enabled: (name: string) => `Включено: «${name}»`,
    disabled: (name: string) => `Отключено: «${name}»`,
    enableRow: 'Включить',
    disableRow: 'Отключить',
    delete: 'Удалить',
    deleting: 'Удаление…',
    deleted: 'Вебхук удалён',
    deleteTitle: 'Удалить вебхук',
    deleteDescPrefix: 'Будет безвозвратно удалён ',
    deleteDescSuffix: '. Действие необратимо.',
    deleteFailed: (name: string) => `Не удалось удалить «${name}»`,
    toggleFailed: (name: string) => `Не удалось обновить «${name}»`,
    newSubscription: 'Новая подписка',
    restarting: 'Перезапуск шлюза…',
    restartNeeded: 'Вебхуки включены, но шлюзу нужен перезапуск, чтобы приёмник заработал.',
    restartGateway: 'Перезапустить шлюз',
    restartingGateway: 'Перезапуск…',
    restartFailed: (detail: string) => `Не удалось перезапустить шлюз${detail}`,
    enabledRestarting: 'Вебхуки включены; шлюз перезапускается…',
    all: '(все)',
    deliverOnly: 'доставлять только',
    createdTitle: 'Подписка создана',
    createdSecretHint: 'Скопируйте секрет сейчас — он показывается один раз.',
    webhookUrl: 'URL вебхука',
    secretOnce: 'Секрет (показывается один раз)',
    done: 'Готово',
    fieldName: 'Имя',
    fieldNamePlaceholder: 'например, github-push',
    fieldDescription: 'Описание',
    fieldDescriptionPlaceholder: 'Что делает этот вебхук (необязательно)',
    fieldEvents: 'События',
    fieldEventsPlaceholder: 'через запятую; пусто — все',
    fieldSkills: 'Навыки',
    fieldSkillsPlaceholder: 'названия навыков через запятую (необязательно)',
    fieldDeliver: 'Доставлять в',
    fieldDeliverOnly: 'Доставлять только данные',
    fieldPrompt: 'Промпт',
    fieldPromptPlaceholder: 'Инструкции агенту при срабатывании вебхука (необязательно)',
    nameRequired: 'Требуется имя',
    create: 'Создать',
    creating: 'Создание…',
    created: 'Создано',
    createFailed: (detail: string) => `Не удалось создать: ${detail}`,
    copy: 'Копировать',
    deliverOptions: {
      log: 'Лог',
      telegram: 'Telegram',
      discord: 'Discord',
      slack: 'Slack',
      email: 'Электронная почта',
      github_comment: 'Комментарий GitHub'
    }
  },
  profiles: {
    close: 'Закрыть профили',
    nameHint: 'Строчные буквы, цифры, дефисы и подчёркивания. Должно начинаться с буквы или цифры.',
    title: 'Профили',
    count: count =>
      countRu(count, {
        one: 'профиль',
        few: 'профиля',
        many: 'профилей'
      }),
    search: 'Поиск профилей...',
    loading: 'Загрузка профилей...',
    newProfile: 'Новый профиль',
    allProfiles: 'Все профили',
    showAllProfiles: 'Показать все профили',
    switchToProfile: name => `Перейти к ${name}`,
    manageProfiles: 'Управление профилями…',
    actionsFor: name => `Действия для ${name}`,
    color: 'Цвет…',
    colorFor: name => `Цвет для ${name}`,
    setColor: color => `Установить цвет ${color}`,
    autoColor: 'Авто',
    noProfiles: 'Профилей пока нет.',
    selectPrompt: 'Выберите профиль для просмотра деталей.',
    refresh: 'Обновить профили',
    refreshing: 'Обновление профилей',
    default: 'по умолчанию',
    skills: count =>
      countRu(count, {
        one: 'навык',
        few: 'навыка',
        many: 'навыков'
      }),
    env: 'Окружение',
    defaultBadge: 'По умолчанию',
    rename: 'Переименовать',
    renameMenu: 'Переименовать…',
    editSoul: 'Редактировать SOUL.md…',
    copySetup: 'Копировать команду',
    copying: 'Копирование...',
    modelLabel: 'Модель',
    skillsLabel: 'Навыки',
    notSet: 'Не задано',
    soulDesc: 'Системный промпт и инструкции персоны этого профиля.',
    soulOptional: 'необязательно',
    soulPlaceholder: mode =>
      `Системный промпт и описание персоны этого профиля.\nОставьте поле пустым, чтобы сохранить ${mode}.`,
    soulPlaceholderCloned: 'клонированный вариант по умолчанию',
    soulPlaceholderEmpty: 'пустой вариант по умолчанию',
    unsavedChanges: 'Несохранённые изменения',
    loadingSoul: 'Загрузка SOUL.md...',
    emptySoul: 'Пустой SOUL.md — начните писать персону...',
    saving: 'Сохранение...',
    saveSoul: 'Сохранить SOUL.md',
    deleteTitle: 'Удалить профиль?',
    deleteDescPrefix: 'Это удалит ',
    deleteDescMid: ' и удалит его каталог ',
    deleteDescSuffix: '. Это действие необратимо.',
    deleting: 'Удаление...',
    createDesc: 'Профили — независимые окружения Hermes: отдельная конфигурация, навыки и SOUL.md.',
    nameLabel: 'Имя',
    cloneFrom: 'Клонировать из',
    cloneFromNone: 'Нет (пустой)',
    cloneFromDesc: 'Копирует конфигурацию, навыки и SOUL.md из выбранного профиля-источника.',
    cloneFromDefault: 'Клонировать из профиля по умолчанию',
    cloneFromDefaultDesc: 'Копирует конфигурацию, навыки и SOUL.md из профиля по умолчанию.',
    invalidName: hint => `Неверное имя. ${hint}`,
    nameRequired: 'Имя обязательно.',
    creating: 'Создание...',
    createAction: 'Создать профиль',
    renameTitle: 'Переименовать профиль',
    renameDescPrefix: 'Переименование обновляет каталог профиля и все скрипты-обёртки в ',
    renameDescSuffix: '.',
    newNameLabel: 'Новое имя',
    renaming: 'Переименование...',
    created: 'Профиль создан',
    renamed: 'Профиль переименован',
    deleted: 'Профиль удалён',
    setupCopied: 'Команда установки скопирована',
    soulSaved: 'SOUL.md сохранён',
    failedLoad: 'Не удалось загрузить профили',
    failedDelete: 'Не удалось удалить профиль',
    failedCopy: 'Не удалось скопировать команду установки',
    failedLoadSoul: 'Не удалось загрузить SOUL.md',
    failedSaveSoul: 'Не удалось сохранить SOUL.md',
    failedCreate: 'Не удалось создать профиль',
    failedRename: 'Не удалось переименовать профиль'
  },
  cron: {
    close: 'Закрыть cron',
    title: 'Расписания',
    count: count =>
      countRu(count, {
        one: 'задание',
        few: 'задания',
        many: 'заданий'
      }),
    search: 'Поиск заданий cron...',
    loading: 'Загрузка заданий cron...',
    states: {
      enabled: 'включено',
      scheduled: 'запланировано',
      running: 'выполняется',
      paused: 'приостановлено',
      disabled: 'отключено',
      error: 'ошибка',
      completed: 'завершено'
    },
    deliveryLabels: {
      local: 'Этот десктоп',
      telegram: 'Telegram',
      discord: 'Discord',
      slack: 'Slack',
      email: 'Эл. почта'
    },
    scheduleLabels: {
      daily: 'Ежедневно',
      weekdays: 'По будням',
      weekly: 'Еженедельно',
      monthly: 'Ежемесячно',
      hourly: 'Ежечасно',
      'every-15-minutes': 'Каждые 15 минут',
      custom: 'Другое'
    },
    scheduleHints: {
      daily: 'Каждый день в 9:00',
      weekdays: 'Понедельник–пятница в 9:00',
      weekly: 'Каждый понедельник в 9:00',
      monthly: 'Первого числа каждого месяца в 9:00',
      hourly: 'В начале каждого часа',
      'every-15-minutes': 'Каждые 15 минут',
      custom: 'Синтаксис cron или естественный язык'
    },
    days: {
      '0': 'Воскресенье',
      '1': 'Понедельник',
      '2': 'Вторник',
      '3': 'Среда',
      '4': 'Четверг',
      '5': 'Пятница',
      '6': 'Суббота',
      '7': 'Воскресенье'
    },
    dayFallback: value => `день ${value}`,
    everyDayAt: time => `Каждый день в ${time}`,
    weekdaysAt: time => `По будням в ${time}`,
    everyDayOfWeekAt: (day, time) => `Каждую неделю: ${day}, ${time}`,
    monthlyOnDayAt: (dayOfMonth, time) => `Ежемесячно ${dayOfMonth} числа в ${time}`,
    topOfHour: 'В начале каждого часа',
    everyHourAt: minute => `Каждый час в :${minute}`,
    newCron: 'Новый cron',
    emptyDescNew:
      'Запланируйте промпт на выполнение по cron-выражению. Hermes выполнит его и доставит результат в выбранный пункт назначения.',
    emptyDescSearch: 'Попробуйте более широкий запрос.',
    emptyTitleNew: 'Заданий пока нет',
    emptyTitleSearch: 'Совпадений нет',
    last: 'Последний:',
    next: 'Следующий:',
    noRuns: 'Запусков ещё не было',
    manage: 'Управление',
    showRuns: 'Показать запуски',
    hideRuns: 'Скрыть запуски',
    runHistory: 'История запусков',
    actionsFor: title => `Действия для ${title}`,
    actionsTitle: 'Действия cron-задания',
    resume: 'Возобновить cron',
    pause: 'Приостановить cron',
    resumeTitle: 'Возобновить',
    pauseTitle: 'Приостановить',
    triggerNow: 'Запустить сейчас',
    edit: 'Изменить cron',
    deleteTitle: 'Удалить задание cron?',
    deleteDescPrefix: 'Это удалит ',
    deleteDescSuffix: ' навсегда. Оно перестанет срабатывать немедленно.',
    deleting: 'Удаление...',
    resumed: 'Cron возобновлён',
    paused: 'Cron приостановлен',
    triggered: 'Cron запущен',
    deleted: 'Cron удалён',
    created: 'Cron создан',
    updated: 'Cron обновлён',
    failedLoad: 'Не удалось загрузить задания cron',
    failedUpdate: 'Не удалось обновить задание cron',
    failedTrigger: 'Не удалось запустить задание cron',
    failedDelete: 'Не удалось удалить задание cron',
    failedSave: 'Не удалось сохранить задание cron',
    editTitle: 'Изменить задание cron',
    createTitle: 'Новое задание cron',
    editDesc: 'Обновите расписание, промпт или пункт доставки. Изменения применятся при следующем запуске.',
    createDesc:
      'Запланируйте промпт на автоматический запуск. Используйте cron-синтаксис или фразу вроде «каждые 15 минут».',
    nameLabel: 'Имя',
    namePlaceholder: 'Утренняя сводка',
    promptLabel: 'Промпт',
    promptPlaceholder: 'Сделай сводку непрочитанных тредов Slack и отправь пять главных по электронной почте…',
    frequencyLabel: 'Частота',
    deliverLabel: 'Доставить в',
    deliverNeedsHomeChannel: 'сначала задайте домашний канал',
    modelLabel: 'Модель',
    modelDefault: 'По умолчанию (глобальная модель)',
    customScheduleLabel: 'Своё расписание',
    customPlaceholder: '0 9 * * * или weekdays at 9am',
    customHint: 'Cron-выражение или фразы вроде «every hour» или «weekdays at 9am».',
    optional: 'Необязательно',
    promptRequired: 'Необходимо указать промпт.',
    promptScheduleRequired: 'Промпт и расписание обязательны.',
    scheduleRequired: 'Необходимо указать расписание.',
    scriptOnlyEditHint: 'Задача только со скриптом (без промпта ИИ). ID задачи:',
    saveChanges: 'Сохранить изменения',
    createAction: 'Создать cron',
    tabs: {
      jobs: 'Задачи',
      blueprints: 'Шаблоны'
    },
    blueprints: {
      tab: 'Шаблоны',
      startFrom: 'Начать с',
      custom: 'Свой вариант',
      subtitle: 'Готовые автоматизации',
      dialogDesc: 'Заполните детали и запланируйте.',
      scheduleIt: 'Запланировать',
      scheduling: 'Планирование…',
      scheduled: 'Шаблон запланирован',
      loading: 'Загрузка шаблонов…',
      failedLoad: 'Не удалось загрузить шаблоны',
      emptyTitle: 'Шаблонов нет',
      emptyDesc: 'На этом бэкенде нет доступных шаблонов автоматизации.'
    }
  },
  artifacts: {
    search: 'Поиск артефактов...',
    refresh: 'Обновить артефакты',
    refreshing: 'Обновление артефактов',
    indexing: 'Индексация недавних артефактов сессий',
    tabAll: 'Все',
    tabImages: 'Изображения',
    tabFiles: 'Файлы',
    tabLinks: 'Ссылки',
    noArtifactsTitle: 'Артефакты не найдены',
    noArtifactsDesc: 'Сгенерированные изображения и файлы появятся здесь по мере создания сессиями.',
    failedLoad: 'Не удалось загрузить артефакты',
    openFailed: 'Не удалось открыть',
    itemsImage: 'изображений',
    itemsLink: 'ссылок',
    itemsFile: 'файлов',
    itemsGeneric: 'элементов',
    zero: '0',
    rangeOf: (start, end, total) => `${start}–${end} из ${total}`,
    goToPage: (itemLabel, page) => `${itemLabel}: перейти на страницу ${page}`,
    colTitleLink: 'Заголовок ссылки',
    colTitleFile: 'Имя',
    colTitleDefault: 'Заголовок / имя',
    colLocationLink: 'URL',
    colLocationFile: 'Путь',
    colLocationDefault: 'Расположение',
    colSession: 'Сессия',
    kindImage: 'изображение',
    kindFile: 'файл',
    kindLink: 'ссылка',
    chat: 'Чат',
    copyUrl: 'Копировать URL',
    copyPath: 'Копировать путь'
  },
  artifactCard: {
    kind: { code: 'Код', html: 'Интерактивная страница', svg: 'Графика' },
    generating: lines =>
      `Создание… ${countRu(lines, {
        one: 'строка',
        few: 'строки',
        many: 'строк'
      })}`,
    versionBadge: count =>
      countRu(count, {
        one: 'версия',
        few: 'версии',
        many: 'версий'
      }),
    open: 'Открыть'
  },
  artifactPane: {
    tabFallback: 'Артефакт',
    modePreview: 'ПРЕДПРОСМОТР',
    modeSource: 'ИСХОДНИК',
    versionOf: (current, total) => `v${current} из ${total}`,
    olderVersion: 'Более старая версия',
    newerVersion: 'Более новая версия',
    latest: 'Последняя',
    copyContent: 'Копировать содержимое',
    download: 'Скачать',
    openInBrowser: 'Открыть в браузере',
    openInBrowserFailed: 'Не удалось открыть в браузере',
    missingTitle: 'Артефакт недоступен',
    missingBody: 'Этого артефакта больше нет в локальном реестре.'
  },
  sidebar: {
    nav: {
      'new-session': 'Новая сессия',
      skills: 'Навыки',
      messaging: 'Мессенджеры',
      artifacts: 'Артефакты'
    },
    searchAria: 'Поиск сессий',
    searchPlaceholder: 'Поиск сессий…',
    clearSearch: 'Очистить поиск',
    noMatch: query => `Нет сессий, соответствующих «${query}».`,
    results: 'Результаты',
    pinned: 'Закреплённые',
    sessions: 'Сессии',
    cronJobs: 'Задания cron',
    groupAriaGrouped: 'Показать сессии одним списком',
    groupAriaUngrouped: 'Сгруппировать сессии по рабочему пространству',
    showProjects: 'Показать проекты',
    showSessions: 'Показать сессии',
    groupTitleGrouped: 'Разгруппировать сессии',
    groupTitleUngrouped: 'Группировать по рабочему пространству',
    allPinned: 'Здесь всё закреплено. Открепите чат, чтобы показать его в недавних.',
    shiftClickHint: 'Shift+клик по чату для закрепления',
    noWorkspace: 'Без рабочего пространства',
    noProject: 'Без проекта',
    projectEmpty: 'Сессий пока нет',
    noSessions: 'Сессий пока нет',
    projects: {
      sectionLabel: 'Проекты',
      newButton: 'Новый проект',
      createTitle: 'Новый проект',
      createDesc: 'Назовите рабочее пространство и добавьте одну или несколько папок.',
      renameTitle: 'Переименовать проект',
      addFolderTitle: 'Добавить папку',
      namePlaceholder: 'напр. Skunkworks',
      foldersLabel: 'Папки',
      ideaLabel: 'Идея',
      ideaPlaceholder: 'О чём этот проект? (сохранится в IDEA.md)',
      ideaGenerate: 'Сгенерировать идею',
      ideaGenerating: 'Генерация…',
      ideaShuffle: 'Перемешать шаблоны',
      noFolders: 'Папок пока не добавлено.',
      addFolder: 'Добавить папку',
      primaryBadge: 'основной',
      removeFolder: 'Удалить',
      create: 'Создать',
      menu: 'Действия проекта',
      menuRename: 'Переименовать',
      menuAppearance: 'Внешний вид',
      noColor: 'Без цвета',
      menuAddFolder: 'Добавить папку',
      menuSetActive: 'Сделать активным',
      menuDelete: 'Удалить',
      reveal: 'Показать в папке',
      copyPath: 'Копировать путь',
      removeFromSidebar: 'Скрыть из боковой панели',
      createFailed: 'Не удалось создать проект',
      staleBackend:
        'Обновите бэкенд Hermes для создания проектов — ваш бэкенд старше этого десктопа (Настройки → Обновления → Бэкенд).',
      deleteConfirm:
        'Это удалит сохранённый проект из Hermes. Файлы, git-репозитории и рабочие деревья останутся без изменений.',
      startWork: 'Новое рабочее дерево',
      newWorktreeTitle: 'Новое рабочее дерево',
      newWorktreeDesc: 'Назовите ветвь для нового рабочего дерева.',
      branchPlaceholder: 'напр. my-feature',
      branchOff: () => ({ after: '', before: 'создать ветку от ' }),
      baseBranchPlaceholder: 'Поиск веток…',
      baseBranchNone: 'Ветки не найдены',
      startWorkFailed: 'Не удалось создать рабочее дерево',
      convertBranch: 'Преобразовать ветвь…',
      convertBranchTitle: 'Преобразовать ветвь',
      convertBranchDesc:
        'Откройте ветви, уже открытые в рабочих деревьях, или создайте рабочее дерево для свободной ветви.',
      convertBranchPlaceholder: 'Поиск ветвей…',
      convertBranchInstead: 'Преобразовать существующую ветвь',
      branchOpenExisting: 'открыть',
      branchSwitchHome: 'переключить основное рабочее дерево',
      branchCreateWorktree: 'новое рабочее дерево',
      branchesLoading: 'Загрузка ветвей…',
      noBranches: 'Ветви не найдены',
      removeWorktree: 'Удалить рабочее дерево',
      removeWorktreeFailed: 'Не удалось удалить рабочее дерево. Возможно, есть незафиксированные изменения.',
      removeWorktreeConfirm:
        'Можно удалить рабочее дерево из git вместе с его каталогом, сохранив ветвь, или только скрыть его из боковой панели и оставить на диске.',
      removeWorktreeDirty:
        'В этом рабочем дереве есть незафиксированные изменения. Можно принудительно удалить его вместе с изменениями или скрыть из боковой панели и оставить на диске.',
      forceRemove: 'Принудительно удалить',
      enter: label => `Открыть ${label}`,
      reorder: label => `Переупорядочить ${label}`,
      toggle: label => `Скрыть/показать сессии ${label}`,
      back: 'Все проекты'
    },
    newSessionIn: label => `Новая сессия в ${label}`,
    showMoreIn: (count, label) => `Показать ещё ${count} в ${label}`,
    loading: 'Загрузка…',
    loadMore: 'Загрузить ещё',
    loadCount: step => `Загрузить ещё ${step}`,
    row: {
      pin: 'Закрепить',
      unpin: 'Открепить',
      copyId: 'Копировать ID',
      export: 'Экспорт',
      branchFrom: 'Ветвь',
      rename: 'Переименовать',
      archive: 'Архивировать',
      newWindow: 'Новое окно',
      hideTabBar: 'Скрыть панель вкладок',
      openInNewTab: 'Открыть в новой вкладке',
      openInSplit: 'Открыть в разделённом представлении',
      copyIdFailed: 'Не удалось скопировать ID сессии',
      actionsFor: title => `Действия для ${title}`,
      sessionActions: 'Действия сессии',
      sessionRunning: 'Сессия выполняется',
      needsInput: 'Требуется ваш ввод',
      waitingForAnswer: 'Ожидание вашего ответа',
      finishedUnread: 'Завершено — не прочитано',
      backgroundRunning: 'Фоновая задача выполняется',
      handoffOrigin: platform => `Передано из ${platform}`,
      ownedByProfile: profile => `Профиль: ${profile}`,
      renamed: 'Переименовано',
      renameFailed: 'Не удалось переименовать',
      renameTitle: 'Переименовать сессию',
      renameDesc: 'Дайте этому чату запоминающееся название. Пусто — очистить.',
      untitledPlaceholder: 'Без названия',
      untitledChat: id => `Чат ${id}`,
      ageNow: 'сейчас',
      ageDay: 'д',
      ageHour: 'ч',
      ageMin: 'м'
    },
    dateDivider: {
      today: 'Сегодня ранее',
      yesterday: 'Вчера',
      thisWeek: 'На этой неделе',
      lastWeek: 'На прошлой неделе',
      thisMonth: 'В этом месяце'
    }
  },
  composer: {
    message: 'Сообщение',
    wakingProfile: profile => `Пробуждение ${profile}…`,
    placeholderStarting: 'Запуск Hermes...',
    placeholderReconnecting: 'Переподключение к Hermes…',
    placeholderFollowUp: 'Отправить дополнение',
    newSessionPlaceholders: [
      'Что будем создавать?',
      'Дайте Hermes задачу',
      'О чём вы думаете?',
      'Опишите, что вам нужно',
      'Чем займёмся?',
      'Спросите что угодно',
      'Начните с цели'
    ],
    followUpPlaceholders: [
      'Отправьте дополнение',
      'Добавьте контекст',
      'Уточните запрос',
      'Что дальше?',
      'Продолжайте',
      'Развивайте идею',
      'Скорректируйте или продолжите'
    ],
    startVoice: 'Начать голосовой разговор',
    queueMessage: 'Поставить в очередь',
    steer: 'Направить текущий запуск',
    stop: 'Стоп',
    send: 'Отправить',
    speaking: 'Озвучивание',
    transcribing: 'Транскрибация',
    thinking: 'Размышление',
    muted: 'Без звука',
    listening: 'Слушаю',
    muteMic: 'Выключить микрофон',
    unmuteMic: 'Включить микрофон',
    stopListening: 'Остановить и отправить',
    stopShort: 'Стоп',
    endConversation: 'Завершить голосовой разговор',
    endShort: 'Завершить',
    stopDictation: 'Остановить диктовку',
    transcribingDictation: 'Транскрибация диктовки',
    voiceDictation: 'Голосовая диктовка',
    speakReplies: 'Озвучивать ответы',
    stopSpeakingReplies: 'Прекратить озвучивание',
    lookupLoading: 'Поиск…',
    lookupNoMatches: 'Совпадений нет.',
    lookupTry: 'Попробуйте',
    lookupOr: 'или',
    commonCommands: 'Частые команды',
    hotkeys: 'Горячие клавиши',
    helpFooter: 'открывает полную панель · Backspace закрывает',
    commandDescs: {
      '/help': 'полный список команд + горячие клавиши',
      '/clear': 'начать новую сессию',
      '/resume': 'возобновить прошлую сессию',
      '/details': 'управлять детализацией транскрипта',
      '/copy': 'копировать выделение или последнее сообщение ассистента',
      '/quit': 'выйти из Hermes'
    },
    hotkeyDescs: {
      'composer.mention': 'ссылаться на файлы, папки, URL, git',
      'composer.slash': 'палитра слэш-команд',
      'composer.help': 'эта справка (Delete для закрытия)',
      'composer.sendNewline': 'отправить · Shift+Enter для переноса',
      'composer.sendQueued': 'отправить следующий ход из очереди',
      'keybinds.openPanel': 'все горячие клавиши',
      'composer.cancel': 'закрыть всплывающее · отменить запуск',
      'composer.history': 'перебор всплывающих / история'
    },
    attachUrlTitle: 'Прикрепить URL',
    attachUrlDesc: 'Hermes загрузит страницу и включит её как контекст для этого шага.',
    urlPlaceholder: 'https://example.com/post',
    urlHintPre: 'Включите полный URL, например ',
    attach: 'Прикрепить',
    queued: count =>
      `В очереди ${countRu(count, {
        one: 'сообщение',
        few: 'сообщения',
        many: 'сообщений'
      })}`,
    queuedPaused: count =>
      `В очереди ${countRu(count, {
        one: 'сообщение',
        few: 'сообщения',
        many: 'сообщений'
      })} — пауза`,
    attachmentOnly: 'Шаг только с вложениями',
    emptyTurn: 'Пустой шаг',
    attachments: count =>
      countRu(count, {
        one: 'вложение',
        few: 'вложения',
        many: 'вложений'
      }),
    editingInComposer: 'Редактирование в поле ввода',
    editingQueuedInComposer: 'Редактирование сообщения из очереди',
    queueEdit: 'Изменить',
    queueSendNext: 'Следующий',
    queueSend: 'Отправить',
    queueDelete: 'Удалить',
    queueResume: 'Продолжить',
    queueResumeTip: 'Остановлено через Стоп — возобновить отправку очереди',
    queueStuckTitle: 'Сообщение из очереди не отправлено',
    queueStuckBody: 'Ход из очереди не удалось отправить. Он всё ещё в очереди — попробуйте снова.',
    previewUnavailable: 'Предпросмотр недоступен',
    previewLabel: label => `Предпросмотр ${label}`,
    couldNotPreview: label => `Не удалось показать ${label}`,
    removeAttachment: label => `Удалить ${label}`,
    dictating: 'Диктовка',
    preparingAudio: 'Подготовка аудио',
    speakingResponse: 'Озвучивание ответа',
    readingAloud: 'Чтение вслух',
    themeSuggestions: 'Предложения тем десктопа',
    noMatchingThemes: 'Нет подходящих тем.',
    themeTryPre: 'Попробуйте ',
    themeTryPost: '.',
    attachLabel: 'Прикрепить',
    files: 'Файлы…',
    folder: 'Папка…',
    images: 'Изображения…',
    pasteImage: 'Вставить изображение',
    url: 'URL…',
    promptSnippets: 'Сниппеты промптов…',
    tipPre: 'Подсказка: введите ',
    tipPost: ' для ссылки на файлы внутри текста.',
    snippetsTitle: 'Сниппеты промптов',
    snippetsDesc: 'Выберите стартовый промпт для вставки в поле ввода.',
    dropFiles: 'Перетащите файлы для прикрепления',
    dropSession: 'Перетащите для связи чата',
    snippets: {
      codeReview: {
        label: 'Ревью кода',
        description: 'Аудит текущих изменений на регрессии, пропущенные крайние случаи и недостающие тесты.',
        text: 'Пожалуйста, ревьюни это на баги, регрессии и недостающие тесты.'
      },
      implementationPlan: {
        label: 'План реализации',
        description: 'Наметить подход до изменения кода, чтобы дифф оставался сфокусированным.',
        text: 'Пожалуйста, составь краткий план реализации перед изменением кода.'
      },
      explainThis: {
        label: 'Объясни это',
        description: 'Разобрать, как работает выбранный код, и указать ключевые файлы.',
        text: 'Пожалуйста, объясни, как это работает, и укажи ключевые файлы.'
      }
    }
  },
  statusStack: {
    agents: 'Агенты',
    background: count =>
      countRu(count, {
        one: 'фоновая задача',
        few: 'фоновые задачи',
        many: 'фоновых задач'
      }),
    goalActive: 'Цель активна',
    goalDone: 'Цель выполнена',
    goalPaused: 'Цель приостановлена',
    goalWaiting: 'Цель ожидает',
    subagents: count =>
      countRu(count, {
        one: 'субагент',
        few: 'субагента',
        many: 'субагентов'
      }),
    todos: (done, total) => `Задачи ${done}/${total}`,
    running: 'Выполняется',
    stop: 'Стоп',
    dismiss: 'Скрыть',
    exit: code => `код ${code}`,
    coding: {
      title: 'Рабочее дерево',
      noBranch: 'Нет ветви',
      detached: 'отвязан',
      clean: 'Чисто',
      changed: count =>
        countRu(count, {
          one: 'изменение',
          few: 'изменения',
          many: 'изменений'
        }),
      ahead: count =>
        `${countRu(count, {
          one: 'коммит',
          few: 'коммита',
          many: 'коммитов'
        })} впереди`,
      behind: count =>
        `${countRu(count, {
          one: 'коммит',
          few: 'коммита',
          many: 'коммитов'
        })} позади`,
      review: 'Ревью',
      close: 'Закрыть',
      openChanges: 'Открыть изменения',
      openFile: 'Открыть файл',
      stage: 'Индексировать',
      unstage: 'Убрать из индекса',
      stageAll: 'Индексировать все',
      viewAsTree: 'В виде дерева',
      viewAsList: 'В виде списка',
      revert: 'Откатить',
      revertAll: 'Откатить все',
      revertConfirm: 'Отменить изменения этого файла и вернуть к зафиксированному состоянию? Это необратимо.',
      revertAllConfirm: 'Отменить все изменения и вернуть файлы к зафиксированному состоянию? Это необратимо.',
      staged: 'В индексе',
      noChanges: 'Нет изменений',
      notRepo: 'Не git-репозиторий',
      noDiff: 'Нет diff для показа',
      scopeUncommitted: 'Незафиксированные',
      scopeBranch: 'Ветвь',
      scopeLastTurn: 'Последний шаг',
      commit: 'Коммит',
      commitAndPush: 'Зафиксировать и отправить',
      commitPlaceholder: 'Сообщение (⌘↵ для коммита)',
      generateCommitMessage: 'Сгенерировать сообщение коммита',
      stopGenerating: 'Остановить генерацию',
      createPr: 'Создать PR',
      openPr: 'Открыть PR',
      ghMissing: 'Установите GitHub CLI (gh) и войдите для открытия PR',
      agentShip: 'Попросить Hermes открыть PR',
      agentShipPrompt:
        'Проверь текущие изменения, закоммить с понятным conventional-commit сообщением, отправь ветвь и открой pull request.',
      newBranch: 'Новая ветвь',
      branchOffFrom: base => `Новая ветвь от ${base}`,
      switchTo: branch => `Перейти к ${branch}`,
      switchFailed: branch => `Не удалось переключиться на ${branch}`,
      worktrees: 'Рабочие деревья'
    }
  },
  updates: {
    stages: {
      idle: 'Подготовка…',
      prepare: 'Подготовка…',
      fetch: 'Загрузка…',
      pull: 'Почти готово…',
      pydeps: 'Завершение…',
      update: 'Обновление Hermes…',
      rebuild: 'Пересборка десктопного приложения…',
      restart: 'Перезапуск Hermes…',
      done: 'Обновление завершено',
      manual: 'Обновление из терминала',
      guiSkew: 'Обновите десктопное приложение',
      error: 'Обновление приостановлено'
    },
    checking: 'Поиск обновлений…',
    checkFailedTitle: 'Не удалось проверить обновления',
    tryAgain: 'Повторить',
    notAvailableTitle: 'Обновление недоступно',
    unsupportedMessage: 'Эта версия Hermes не может обновить сама себя из приложения.',
    connectionRetry: 'Проверьте подключение и повторите.',
    latestBody: 'У вас последняя версия.',
    latestBodyBackend: 'Бэкенд работает на последней версии.',
    allSetTitle: 'Всё актуально',
    availableTitle: 'Доступно новое обновление',
    availableBody: 'Новая версия Hermes готова к установке.',
    availableTitleBackend: 'Доступно обновление бэкенда',
    availableBodyBackend: 'Новая версия подключённого бэкенда Hermes готова к установке.',
    availableBodyNoChangelog: 'Новая версия готова. Примечания к выпуску недоступны для этого типа установки.',
    updateNow: 'Обновить сейчас',
    maybeLater: 'Напомнить позже',
    moreChanges: count =>
      `+ ещё ${countRu(count, {
        one: 'изменение',
        few: 'изменения',
        many: 'изменений'
      })}.`,
    manualTitle: 'Обновление из терминала',
    manualBody:
      'Вы установили Hermes из командной строки, поэтому обновления тоже запускаются там. Вставьте это в терминал:',
    manualPickedUp: 'Hermes подхватит новую версию при следующем запуске.',
    guiSkewTitle: 'Обновите десктопное приложение',
    guiSkewBody:
      'Бэкенд обновлён, но пакет десктопного приложения не изменён. Обновите или переустановите десктоп Hermes (AppImage / .deb / .rpm) для соответствия.',
    copy: 'Копировать',
    copied: 'Скопировано',
    done: 'Готово',
    applyingBody:
      'Программа обновления Hermes открывается в собственном окне и автоматически перезапускает Hermes после завершения. Не открывайте Hermes сами во время обновления.',
    applyingBodyBackend:
      'Удалённый бэкенд применяет обновление и перезапустится. Hermes переподключится автоматически.',
    applyingClose: 'Это окно закроется во время обновления, затем Hermes откроется сам.',
    errorTitle: 'Обновление не завершено',
    errorBody: 'Ничего страшного — ничего не потеряно. Можете попробовать снова.',
    notNow: 'Не сейчас',
    applyStatus: {
      preparing: 'Обновление бэкенда…',
      pulling: 'Бэкенд обновляется…',
      restarting: 'Бэкенд перезапускается для применения обновления…',
      notAvailable: 'Обновление недоступно для этого бэкенда.',
      failed: 'Сбой обновления бэкенда.',
      noReturn: 'Бэкенд не вернулся в сеть. Обновление могло не завершиться — проверьте хост бэкенда.'
    }
  },
  install: {
    stageStates: {
      pending: 'Ожидание',
      running: 'Установка',
      succeeded: 'Готово',
      skipped: 'Пропущено',
      failed: 'Ошибка'
    },
    oneTimeTitle: 'Hermes требует одноразовой установки',
    unsupportedDesc: platform =>
      `Автоматическая установка при первом запуске пока недоступна на ${platform}. Откройте терминал и выполните команду ниже, затем перезапустите приложение. Последующие запуски пропустят этот шаг.`,
    installCommand: 'Команда установки',
    copyCommand: 'Копировать команду',
    viewDocs: 'Открыть документацию',
    installTo: 'Установка в',
    retryAfterRun: 'Я выполнил — повторить',
    setupChoiceTitle: 'Настройка Hermes Desktop',
    setupChoiceDesc:
      'Подключите приложение к уже работающему шлюзу Hermes или установите Hermes локально на этот компьютер.',
    connectExistingTitle: 'Подключиться к существующему Hermes',
    connectExistingShort: 'Подключить существующий',
    connectExistingDesc:
      'Использовать удалённый бэкенд с токеном сессии или входом через браузер. Локальная установка не запустится.',
    installLocalTitle: 'Установить Hermes локально',
    installLocalDesc: 'Скачать Hermes, создать Python-окружение и запустить бэкенд на этом компьютере.',
    localStartUnavailable: 'Локальная установка не смогла начаться. Перезапустите Hermes Desktop и повторите.',
    remoteSetupTitle: 'Подключение к существующему Hermes',
    remoteSetupDesc: 'Введите URL шлюза. Hermes Desktop определит, нужен токен или вход через браузер.',
    remoteUrlTitle: 'URL шлюза',
    remoteUrlDesc: 'Базовый URL шлюза Hermes, включая https:// для удалённого.',
    remoteUrlPlaceholder: 'https://gateway.example.com/hermes',
    probing: 'Определение аутентификации шлюза…',
    probeError: 'Не удалось достучаться до шлюза Hermes.',
    identityProvider: 'ваш провайдер идентификации',
    authTitle: 'Аутентификация',
    authNeedsOauth: provider => `Войдите через ${provider} перед проверкой этого шлюза.`,
    authSignedIn: 'Вход через браузер завершён.',
    connected: 'Подключено',
    signIn: 'Войти',
    signInWith: provider => `Войти через ${provider}`,
    enterUrlFirst: 'Сначала введите URL шлюза.',
    signInIncomplete: 'Окно входа закрылось до завершения аутентификации.',
    tokenTitle: 'Токен сессии',
    tokenDesc: 'Вставьте токен сессии из .env файла удалённого шлюза.',
    pasteSessionToken: 'Вставьте токен сессии',
    incompleteSignInTest: 'Войдите перед проверкой этого OAuth-шлюза.',
    incompleteTokenTest: 'Введите токен сессии перед проверкой шлюза.',
    testConnection: 'Проверить подключение',
    testSucceeded: (baseUrl, version) => `Подключено к ${baseUrl}${version ? ` (${version})` : ''}.`,
    applyRemote: 'Применить и переподключить',
    backToSetup: 'Назад',
    failedTitle: 'Установка не удалась',
    settingUpTitle: 'Настройка Hermes Agent',
    finishingTitle: 'Завершение',
    failedDesc:
      'Один из этапов установки не удался. На Windows это может произойти, если другой экземпляр Hermes CLI или десктопа уже запущен. Остановите все запущенные экземпляры Hermes и повторите. Проверьте детали ниже или лог десктопа для полной транскрипции.',
    activeDesc:
      'Это одноразовая настройка. Установщик Hermes загружает зависимости и настраивает вашу машину. Последующие запуски пропустят этот шаг.',
    progress: (completed, total) =>
      `${completed} из ${countRu(total, {
        one: 'шага',
        few: 'шагов',
        many: 'шагов'
      })} завершено`,
    currentStage: stage => ` — сейчас: ${stage}`,
    fetchingManifest: 'Получение манифеста установщика...',
    error: 'Ошибка',
    hideOutput: 'Скрыть вывод установщика',
    showOutput: 'Показать вывод установщика',
    lines: count =>
      countRu(count, {
        one: 'строка',
        few: 'строки',
        many: 'строк'
      }),
    noOutput: 'Пока нет вывода.',
    cancelling: 'Отмена...',
    cancelInstall: 'Отменить установку',
    transcriptSaved: 'Полная транскрипция сохранена в',
    copiedOutput: 'Скопировано!',
    copyOutput: 'Копировать вывод',
    reloadRetry: 'Перезагрузить и повторить'
  },
  onboarding: {
    headerTitle: 'Давайте настроим Hermes Agent',
    headerDesc: 'Подключите провайдера моделей, чтобы начать общение. Большинство опций — один клик.',
    preparingInstall: 'Hermes завершает установку. Обычно занимает меньше минуты при первом запуске.',
    starting: 'Запуск Hermes…',
    lookingUpProviders: 'Поиск провайдеров...',
    collapse: 'Свернуть',
    otherProviders: 'Другие провайдеры',
    haveApiKey: 'У меня есть API-ключ',
    chooseLater: 'Выберу провайдера позже',
    recommended: 'Рекомендуется',
    connected: 'Подключено',
    featuredPitch: 'Одна подписка, 300+ передовых моделей — рекомендуемый способ запуска Hermes',
    fireworksPitch: 'Прямой API моделей — передовые модели на Fireworks',
    openRouterPitch: 'Один ключ, сотни моделей — отличный вариант по умолчанию',
    apiKeyOptions: {
      fireworks: {
        short: 'прямой API моделей',
        description: 'Прямой доступ к моделям Fireworks AI.'
      },
      openrouter: {
        short: 'один ключ, много моделей',
        description: 'Сотни моделей за одним ключом. Хороший выбор для новых установок.'
      },
      openai: {
        short: 'Модели GPT',
        description: 'Прямой доступ к моделям OpenAI.'
      },
      gemini: {
        short: 'Модели Gemini',
        description: 'Прямой доступ к моделям Google Gemini.'
      },
      xai: {
        short: 'Модели Grok',
        description: 'Прямой доступ к моделям xAI Grok.'
      },
      local: {
        short: 'собственный сервер',
        description:
          'Подключите Hermes к локальному или самостоятельно размещённому OpenAI-совместимому эндпоинту (vLLM, llama.cpp, Ollama и т. д.).'
      }
    },
    backToSignIn: 'Назад ко входу',
    getKey: 'Получить ключ',
    replaceCurrent: 'Заменить текущее значение',
    pasteApiKey: 'Вставить API-ключ',
    localApiKeyPlaceholder: 'API-ключ (необязательно — только если он требуется вашему эндпоинту)',
    couldNotSave: 'Не удалось сохранить учётные данные.',
    connecting: 'Подключение',
    update: 'Обновить',
    flowSubtitles: {
      pkce: 'Открывает браузер для входа, затем продолжает здесь',
      device_code: 'Открывает страницу верификации в браузере — Hermes подключается автоматически',
      external: 'Войдите один раз в терминале, затем вернитесь в чат'
    },
    startingSignIn: provider => `Запуск входа для ${provider}...`,
    verifyingCode: provider => `Проверка кода с ${provider}...`,
    connectedProvider: provider => `${provider} подключён`,
    connectedPicking: provider => `${provider} подключён. Выбор модели по умолчанию...`,
    signInFailed: 'Вход не удался. Попробуйте снова.',
    pickDifferentProvider: 'Выбрать другого провайдера',
    signInWith: provider => `Войти через ${provider}`,
    openedBrowser: provider => `Мы открыли ${provider} в вашем браузере.`,
    authorizeThere: 'Авторизуйте Hermes там.',
    copyAuthCode: 'Скопируйте код авторизации и вставьте ниже.',
    pasteAuthCode: 'Вставить код авторизации',
    reopenAuthPage: 'Открыть страницу авторизации снова',
    autoBrowser: provider =>
      `Мы открыли ${provider} в вашем браузере. Авторизуйте Hermes там, и вы подключитесь автоматически — ничего копировать не нужно.`,
    reopenSignInPage: 'Открыть страницу входа снова',
    waitingAuthorize: 'Ожидание вашей авторизации...',
    externalPending: provider =>
      `${provider} выполняет вход через собственный CLI. Запустите эту команду в терминале, затем выберите «Я вошёл»:`,
    signedIn: 'Я вошёл',
    deviceCodeOpened: provider => `Мы открыли ${provider} в вашем браузере. Введите этот код там:`,
    reopenVerification: 'Открыть страницу верификации снова',
    copy: 'Копировать',
    defaultModel: 'Модель по умолчанию',
    freeTier: 'Бесплатный уровень',
    pro: 'Pro',
    free: 'Бесплатно',
    price: (input, output) => `${input} вх / ${output} вых за 1 млн токенов`,
    change: 'Изменить',
    startChatting: 'Начать',
    docs: provider => `Документация ${provider}`
  },
  modelPicker: {
    title: 'Сменить модель',
    current: 'текущая:',
    unknown: '(неизвестно)',
    search: 'Фильтр провайдеров и моделей...',
    noModels: 'Модели не найдены.',
    addProvider: 'Добавить провайдера',
    loadFailed: 'Не удалось загрузить модели',
    noAuthenticatedProviders: 'Нет авторизованных провайдеров.',
    pro: 'Pro',
    proNeedsSubscription: 'Модели Pro требуют платную подписку Nous.',
    free: 'Бесплатно',
    freeTier: 'Бесплатный уровень',
    priceTitle: 'Цена ввод / вывод за миллион токенов',
    wasPrice: 'было'
  },
  modelVisibility: {
    title: 'Модели',
    search: 'Поиск моделей',
    noAuthenticatedProviders: 'Нет авторизованных провайдеров.',
    addProvider: 'Добавить провайдера…'
  },
  shell: {
    windowControls: 'Управление окном',
    paneControls: 'Управление панелями',
    appControls: 'Управление приложением',
    modelMenu: {
      search: 'Поиск моделей',
      noModels: 'Модели не найдены',
      editModels: 'Изменить модели…',
      refreshModels: 'Обновить модели',
      fast: 'Быстро'
    },
    modelOptions: {
      noOptions: 'Для этой модели нет опций',
      options: 'Опции',
      thinking: 'Размышление',
      fast: 'Быстро',
      effort: 'Усилие',
      minimal: 'Минимум',
      low: 'Низкое',
      medium: 'Среднее',
      high: 'Высокое',
      xhigh: 'Очень высокий',
      max: 'Макс',
      ultra: 'Ультра',
      updateFailed: 'Не удалось обновить опцию модели',
      fastFailed: 'Не удалось обновить быстрый режим'
    },
    gatewayMenu: {
      gateway: 'Шлюз',
      connected: 'Подключено',
      connecting: 'Подключение',
      offline: 'Офлайн',
      inferenceReady: 'Инференс готов',
      inferenceNotReady: 'Инференс не готов',
      checkingInference: 'Проверка инференса',
      disconnected: 'Отключено',
      openSystem: 'Открыть системную панель',
      connection: label => `Подключение: ${label}`,
      recentActivity: 'Недавняя активность',
      viewAllLogs: 'Все логи →',
      messagingPlatforms: 'Платформы мессенджеров'
    },
    approvalMode: {
      title: 'Режим подтверждения',
      ariaLabel: mode => `Режим подтверждения: ${mode}`,
      manual: 'Ручной',
      manualDescription: 'Запрашивать подтверждение перед действиями, требующими его',
      smart: 'Умный',
      smartDescription: 'Автоматически оценивать действия и запрашивать подтверждение при необходимости',
      off: 'Отключён',
      offDescription: 'Выполнять без запросов на подтверждение'
    },
    statusbar: {
      unknown: 'неизвестно',
      restart: 'перезапуск',
      update: 'обновление',
      updateInProgress: 'Идёт обновление',
      commitsBehind: (count, branch) =>
        `${countRu(count, {
          one: 'коммит',
          few: 'коммита',
          many: 'коммитов'
        })} позади ${branch}`,
      desktopVersion: version => `Hermes Desktop v${version}`,
      backendVersion: version => `Бэкенд v${version}`,
      clientLabel: version => `клиент v${version}`,
      connectionSsh: host => `SSH: ${host}`,
      connectionRemote: host => `Удалённо: ${host}`,
      connectionCloud: host => `Облако: ${host}`,
      connectionCloudTooltip: host => `Подключено к Hermes Cloud (${host}) · нажмите для управления`,
      connectionSshTooltip: host => `Подключено по SSH к ${host} · нажмите для управления`,
      connectionRemoteTooltip: host => `Подключено к удалённому бэкенду ${host} · нажмите для управления`,
      backendLabel: version => `бэкенд v${version}`,
      commit: sha => `коммит ${sha}`,
      branch: branch => `ветвь ${branch}`,
      closeCommandCenter: 'Закрыть командный центр',
      openCommandCenter: 'Открыть командный центр',
      showTerminal: 'Показать терминал',
      hideTerminal: 'Скрыть терминал',
      gateway: 'Шлюз',
      gatewayReady: 'готов',
      gatewayNeedsSetup: 'требует настройки',
      gatewayChecking: 'проверка',
      gatewayConnecting: 'подключение',
      gatewayOffline: 'офлайн',
      gatewayRestarting: 'перезапуск…',
      gatewayTitle: 'Статус шлюза инференса Hermes',
      customizeTitle: 'Показывать в строке состояния',
      toggleApprovalMode: 'Подтверждения',
      toggleBackendVersion: 'Версия бэкенда',
      toggleCommandCenter: 'Командный центр',
      toggleContextUsage: 'Индикатор контекста',
      toggleRunningTimer: 'Таймер шага',
      toggleSessionTimer: 'Таймер сессии',
      toggleTerminal: 'Терминал',
      toggleVersion: 'Версия и обновления',
      toggleWorkspace: 'Рабочее пространство',
      agents: 'Агенты',
      closeAgents: 'Закрыть агентов',
      openAgents: 'Открыть агентов',
      subagents: count =>
        countRu(count, {
          one: 'субагент',
          few: 'субагента',
          many: 'субагентов'
        }),
      failed: count =>
        countRu(count, {
          one: 'с ошибкой',
          few: 'с ошибками',
          many: 'с ошибками'
        }),
      running: count =>
        countRu(count, {
          one: 'выполняется',
          few: 'выполняются',
          many: 'выполняются'
        }),
      cron: 'Cron',
      openCron: 'Открыть задания cron',
      webhooks: 'Вебхуки',
      openWebhooks: 'Открыть вебхуки',
      starmap: 'Граф памяти',
      openStarmap: 'Открыть граф памяти',
      turnRunning: 'Выполняется',
      currentTurnElapsed: 'Время текущего шага',
      contextUsage: 'Использование контекста',
      contextUsagePanel: {
        categories: {
          conversation: 'Диалог',
          mcp: 'MCP',
          memory: 'Память',
          rules: 'Правила',
          skills: 'Навыки',
          subagent_definitions: 'Определения субагентов',
          system_prompt: 'Системный промпт',
          tool_definitions: 'Определения инструментов'
        },
        empty: 'Пока нет данных контекста',
        loading: 'Загрузка разбивки…',
        percentFull: percent => `${percent}% заполнено`,
        title: 'Использование контекста',
        tokenSummary: (used, max) => `Токены: ${used} / ${max}`
      },
      openContextUsage: 'Открыть разбивку использования контекста',
      session: 'Сессия',
      runtimeSessionElapsed: 'Время сессии',
      yoloOn: 'YOLO вкл — авто-одобрение опасных команд. Нажмите для выключения. Shift+клик переключает глобально.',
      yoloOff: 'YOLO выкл — нажмите для авто-одобрения опасных команд. Shift+клик переключает глобально.',
      modelNone: 'нет',
      noModel: 'нет модели',
      switchModel: 'Сменить модель',
      openModelPicker: 'Открыть выбор модели',
      modelPinned: 'закреплена вами; новые чаты используют её вместо модели из настроек',
      modelTitle: (provider, model) => `Модель · ${provider}: ${model}`,
      providerModelTitle: (provider, model) => `${provider} · ${model}`
    }
  },
  rightSidebar: {
    aria: 'Правая панель',
    panelsAria: 'Панели правой стороны',
    files: 'Файловая система',
    terminal: 'Терминал',
    noFolderSelected: 'Папка не выбрана',
    changeCwdTitle: 'Сменить рабочий каталог',
    remotePickerTitle: 'Выбрать удалённую папку',
    remotePickerDescription: 'Просмотр папок на подключённом бэкенде.',
    remotePickerSelect: 'Выбрать папку',
    folderTip: cwd => `${cwd} — нажмите для смены папки`,
    openFolder: 'Открыть папку',
    refreshTree: 'Обновить дерево',
    collapseAll: 'Свернуть все папки',
    previewUnavailable: 'Предпросмотр недоступен',
    couldNotPreview: path => `Не удалось показать ${path}`,
    noProjectTitle: 'Нет проекта',
    noProjectBody: 'Откройте проект для просмотра файлов и рецензирования изменений.',
    noProjectOpen: 'Проект не открыт',
    noDiffs: 'Нет diff',
    unreadableTitle: 'Не читается',
    unreadableBody: error => `Не удалось прочитать эту папку (${error}).`,
    emptyTitle: 'Пусто',
    emptyBody: 'Эта папка пуста.',
    treeErrorTitle: 'Ошибка дерева',
    treeErrorBody: 'Дерево файлов столкнулось с ошибкой при отрисовке этой папки.',
    tryAgain: 'Повторить',
    loadingTree: 'Загрузка дерева файлов',
    loadingFiles: 'Загрузка файлов',
    terminalHide: 'Скрыть терминал',
    terminalsAria: 'Терминалы',
    terminalNew: 'Новый терминал',
    terminalCloseOthers: 'Закрыть остальные',
    terminalCloseAll: 'Закрыть все',
    addToChat: 'Добавить в чат'
  },
  preview: {
    tab: 'Предпросмотр',
    closeTab: label => `Закрыть ${label}`,
    closeOthers: 'Закрыть остальные',
    closeToRight: 'Закрыть справа',
    closeAll: 'Закрыть все',
    closePane: 'Закрыть панель предпросмотра',
    loading: 'Загрузка предпросмотра',
    unavailable: 'Предпросмотр недоступен',
    opening: 'Открытие...',
    hide: 'Скрыть',
    openPreview: 'Открыть предпросмотр',
    openInBrowser: 'Открыть в браузере',
    linkHint: '⌘/Ctrl-клик для панели предпросмотра',
    sourceLineTitle: 'Клик — выбрать · Shift+клик — расширить · перетащите в поле ввода',
    source: 'ИСХОДНИК',
    renderedPreview: 'ПРЕДПРОСМОТР',
    diff: 'DIFF',
    unknownSize: 'неизвестный размер',
    binaryTitle: 'Похоже на бинарный файл',
    binaryBody: label => `Предпросмотр ${label} может показать нечитаемый текст.`,
    largeTitle: 'Этот файл большой',
    largeBody: (label, size) => `${label} — ${size}. Hermes покажет только первые 512 КБ.`,
    previewAnyway: 'Показать всё равно',
    truncated: 'Показаны первые 512 КБ.',
    noInlineTitle: 'Нет встроенного предпросмотра',
    noInlineBody: mimeType => `${mimeType || 'Этот тип файла'} всё ещё можно прикрепить как контекст.`,
    edit: 'Изменить',
    editing: 'Редактирование',
    unsavedChanges: 'Несохранённые изменения',
    saveFailed: message => `Не удалось сохранить: ${message}`,
    diskChangedTitle: 'Файл изменён на диске',
    diskChangedBody:
      'Этот файл изменился с момента открытия. Перезаписать вашей версией или отбросить правки и перезагрузить?',
    overwrite: 'Перезаписать',
    discardReload: 'Отбросить и перезагрузить',
    console: {
      deselect: 'Снять выделение',
      select: 'Выбрать запись',
      copyFailed: 'Не удалось скопировать вывод консоли',
      copyEntry: 'Копировать запись',
      sendEntry: 'Отправить запись в чат',
      messages: count =>
        `${countRu(count, {
          one: 'сообщение',
          few: 'сообщения',
          many: 'сообщений'
        })} консоли`,
      resize: 'Изменить размер консоли предпросмотра',
      title: 'Консоль предпросмотра',
      selected: count =>
        pluralRu(count, {
          one: `Выбрана ${count} запись`,
          few: `Выбраны ${count} записи`,
          many: `Выбрано ${count} записей`
        }),
      sendToChat: 'Отправить в чат',
      copySelected: 'Копировать выделенное в буфер',
      copyAll: 'Копировать всё в буфер',
      copy: 'Копировать',
      clear: 'Очистить',
      empty: 'Сообщений консоли пока нет.',
      promptHeader: 'Консоль предпросмотра:',
      sentTitle: 'Отправлено в чат',
      sentMessage: count =>
        `${countRu(count, {
          one: 'запись лога добавлена',
          few: 'записи лога добавлены',
          many: 'записей лога добавлено'
        })} в поле ввода`
    },
    web: {
      appFailedToBoot: 'Не удалось запустить приложение предпросмотра',
      serverNotFound: 'Сервер не найден',
      failedToLoad: 'Предпросмотр не загрузился',
      tryAgain: 'Повторить',
      restarting: 'Hermes перезапускается...',
      askRestart: 'Попросить Hermes перезапустить сервер',
      lookingRestart: taskId => `Hermes ищет сервер предпросмотра для перезапуска (${taskId})`,
      restartingTitle: 'Перезапуск сервера предпросмотра',
      restartingMessage: 'Hermes работает в фоне. Следите за консолью предпросмотра.',
      startRestartFailed: message => `Не удалось запустить перезапуск: ${message}`,
      restartFailed: 'Перезапуск сервера не удался',
      hideConsole: 'Скрыть консоль предпросмотра',
      showConsole: 'Показать консоль предпросмотра',
      hideDevTools: 'Скрыть DevTools предпросмотра',
      openDevTools: 'Открыть DevTools предпросмотра',
      finishedRestarting: message => `Hermes завершил перезапуск сервера предпросмотра${message ? `: ${message}` : ''}`,
      failedRestarting: message => `Перезапуск не удался: ${message}`,
      unknownError: 'неизвестная ошибка',
      restartedTitle: 'Сервер предпросмотра перезапущен',
      reloadingNow: 'Перезагрузка предпросмотра.',
      restartFailedTitle: 'Перезапуск предпросмотра не удался',
      restartFailedMessage: 'Hermes не смог перезапустить сервер.',
      stillWorking:
        'Hermes всё ещё работает, но результат перезапуска ещё не получен. Возможно, команда сервера выполняется на переднем плане.',
      workspaceReloading: 'Рабочее пространство изменилось, перезагрузка предпросмотра',
      fileChanged: url => `Файл изменился, перезагрузка предпросмотра: ${url}`,
      filesChanged: (count, url) =>
        `${countRu(count, {
          one: 'изменение файла',
          few: 'изменения файлов',
          many: 'изменений файлов'
        })}, перезагрузка предпросмотра: ${url}`,
      watchFailed: message => `Не удалось отслеживать файл: ${message}`,
      moduleMimeDescription:
        'Модульные скрипты обслуживаются с неверным MIME-типом. Обычно это значит, что статический файловый сервер обслуживает Vite/React-приложение вместо dev-сервера проекта.',
      loadFailedConsole: (code, message) => `Сбой загрузки${code ? ` (${code})` : ''}: ${message}`,
      unreachableDescription: 'Страницу предпросмотра не удалось открыть.',
      openTarget: url => `Открыть ${url}`,
      fallbackTitle: 'Предпросмотр'
    }
  },
  zones: {
    showHeader: 'Показать заголовок',
    hideHeader: 'Скрыть заголовок',
    minimize: 'Свернуть',
    restore: 'Восстановить',
    closeRunningTitle: 'Закрыть вкладку с запущенным процессом?',
    closeRunningBody:
      'Этот чат всё ещё работает или ждёт вашего ответа. Закрытие вкладки только скроет её — сессия продолжит работу, и её можно будет снова открыть из боковой панели.',
    closeRunningConfirm: 'Закрыть вкладку',
    closeOthers: 'Закрыть остальные',
    closeToRight: 'Закрыть вкладки справа',
    closeAll: 'Закрыть все',
    newSessionTab: 'Новая вкладка сессии',
    split: dir => `Разделить ${dir}`,
    move: dir => `Переместить ${dir}`,
    dirUp: 'вверх',
    dirDown: 'вниз',
    dirLeft: 'влево',
    dirRight: 'вправо',
    pluginDisabled: pluginId => `Плагин «${pluginId}» отключён`,
    pluginDisabledBody: 'Снова включите его в «Настройки → Плагины», чтобы вернуть панель.',
    missingPane: paneId => `отсутствующая панель: ${paneId}`,
    editTitle: 'Макеты',
    editHint:
      'Выберите макет или перетаскивайте панели между зонами. Щёлкните правой кнопкой по зоне, чтобы разделить её.',
    reset: 'Сбросить',
    templates: 'Шаблоны',
    custom: 'Пользовательский',
    newGridLayout: 'Новый сеточный макет',
    saveCurrentAs: 'Сохранить текущую компоновку как шаблон',
    nameLayoutPlaceholder: 'Назовите этот макет…',
    deletePreset: name => `Удалить ${name}`,
    zoneEditorTitle: 'Редактор зон',
    editorHintPre: 'щёлкните, чтобы разделить · ',
    editorHintPost:
      ' меняет направление линии · перетащите через зоны, чтобы объединить · перетаскивайте общие границы для изменения размера',
    templateColumns: 'Столбцы',
    templateRows: 'Строки',
    templateGrid: 'Сетка',
    templatePriority: 'Приоритет',
    zoneTag: index => `зона ${index}`,
    mergeZones: count =>
      `Объединить ${countRu(count, {
        one: 'зону',
        few: 'зоны',
        many: 'зон'
      })}`,
    customZoneName: count =>
      `Пользовательский макет (${countRu(count, {
        one: 'зона',
        few: 'зоны',
        many: 'зон'
      })})`,
    layoutNamePlaceholder: fallback => `Название макета (${fallback})`,
    saveApply: 'Сохранить и применить',
    notExpressible: 'эта компоновка образует «вертушку» и пока не может быть выражена вложенными разделениями',
    zoneCount: count =>
      countRu(count, {
        one: 'зона',
        few: 'зоны',
        many: 'зон'
      })
  },
  assistant: {
    thread: {
      loadingSession: 'Загрузка сессии',
      showEarlier: 'Показать ранние сообщения',
      loadingResponse: 'Hermes загружает ответ',
      resumeWhenBackgroundDone: count =>
        `Продолжит после завершения ${countRu(count, {
          one: 'фоновой задачи',
          few: 'фоновых задач',
          many: 'фоновых задач'
        })}`,
      thinking: 'Размышление',
      today: time => `Сегодня, ${time}`,
      yesterday: time => `Вчера, ${time}`,
      copy: 'Копировать',
      refresh: 'Обновить',
      moreActions: 'Ещё действия',
      branchNewChat: 'Ветвь в новом чате',
      dismissError: 'Скрыть ошибку',
      readAloudFailed: 'Не удалось озвучить',
      preparingAudio: 'Подготовка аудио...',
      stopReading: 'Остановить чтение',
      readAloud: 'Озвучить',
      editMessage: 'Редактировать сообщение',
      expandMessage: 'Развернуть сообщение',
      scrollToBottom: 'Прокрутить вниз',
      stop: 'Стоп',
      restorePrevious: 'Восстановить предыдущую контрольную точку',
      restoreCheckpoint: 'Восстановить контрольную точку',
      restoreFromHere: 'Восстановить контрольную точку — перезапуск от этого промпта',
      restoreTitle: 'Восстановить до этой контрольной точки?',
      restoreBody: 'Всё после этого промпта удаляется из диалога, и промпт запускается заново отсюда.',
      restoreConfirm: 'Восстановить и перезапустить',
      restoreNext: 'Восстановить следующую контрольную точку',
      goForward: 'Вперёд',
      sendEdited: 'Отправить изменённое сообщение',
      attachingFile: 'Прикрепление…'
    },
    approval: {
      gatewayDisconnected: 'Шлюз Hermes не подключён',
      sendFailed: 'Не удалось отправить ответ подтверждения',
      run: 'Выполнить',
      command: 'Команда',
      moreOptions: 'Больше опций подтверждения',
      allowSession: 'Разрешить для этой сессии',
      alwaysAllowMenu: 'Всегда разрешать…',
      jumpToApproval: 'Требуется подтверждение',
      reject: 'Отклонить',
      alwaysTitle: 'Всегда разрешать эту команду?',
      alwaysDescription: pattern =>
        `Это добавит шаблон «${pattern}» в ваш постоянный список разрешений (~/.hermes/config.yaml). Hermes больше не будет спрашивать для подобных команд — в этой или любой будущей сессии.`,
      alwaysAllow: 'Всегда разрешать'
    },
    clarify: {
      notReady: 'Запрос уточнения ещё не готов',
      gatewayDisconnected: 'Шлюз Hermes не подключён',
      sendFailed: 'Не удалось отправить ответ уточнения',
      loadingQuestion: 'Загрузка вопроса…',
      other: 'Другое (введите ответ)',
      placeholder: 'Введите ответ…',
      skip: 'Пропустить',
      skipped: 'Пропущено',
      continueLabel: 'Продолжить',
      lateAnswer: (question, choice) => `Re: «${question}» — мой ответ: ${choice}`,
      lateAnswerTip: 'Черновик ответа как следующее сообщение',
      lateAnswerHint:
        'Этот запрос больше не ждёт. Выберите вариант, чтобы добавить ответ в черновик следующего сообщения.'
    },
    tool: {
      code: 'Код',
      copyCode: 'Копировать код',
      renderingImage: 'Рендеринг изображения',
      copyOutput: 'Копировать вывод',
      copyCommand: 'Копировать команду',
      copyContent: 'Копировать содержимое',
      copyUrl: 'Копировать URL',
      copyResults: 'Копировать результаты',
      copyQuery: 'Копировать запрос',
      copyFile: 'Копировать файл',
      copyPath: 'Копировать путь',
      outputAlt: 'Вывод инструмента',
      rawResponse: 'Сырой ответ',
      copyActivity: 'Копировать активность',
      recoveredOne: 'Восстановлено после 1 сбойного шага',
      recoveredMany: count =>
        `Восстановлено после ${countRu(count, {
          one: 'сбойного шага',
          few: 'сбойных шагов',
          many: 'сбойных шагов'
        })}`,
      failedOne: '1 шаг не удался',
      failedMany: count =>
        countRu(count, {
          one: 'шаг не удался',
          few: 'шага не удались',
          many: 'шагов не удалось'
        }),
      statusRunning: 'Выполняется',
      statusError: 'Ошибка',
      statusRecovered: 'Восстановлено',
      statusDone: 'Готово',
      actions: {
        read: 'Чтение',
        reading: 'Чтение',
        opened: 'Открыт',
        opening: 'Открытие',
        failedToOpen: 'Не удалось открыть',
        searched: 'Найдено',
        searching: 'Поиск',
        ran: 'Выполнено',
        running: 'Выполнение',
        ranCode: 'Код выполнен',
        runningCode: 'Выполнение скрипта'
      },
      prefixes: {
        browser: 'Браузер',
        web: 'Веб'
      },
      titleTemplates: {
        actionCommand: (action, command) => `${action} ${command}`,
        actionQuoted: (action, value) => `${action} «${value}»`,
        actionTarget: (action, target) => `${action} ${target}`,
        prefixedDone: (prefix, action) => `${prefix} ${action}`,
        runningPrefixedTool: (prefix, action) => `${action.toLowerCase()} (${prefix.toLowerCase()})`,
        runningTool: action => `${action.toLowerCase()}`
      },
      titles: {
        browser_click: {
          done: 'Кликнул элемент страницы',
          pending: 'Клик по элементу страницы',
          pendingAction: 'Клик'
        },
        browser_fill: {
          done: 'Заполнил поле формы',
          pending: 'Заполнение поля формы',
          pendingAction: 'Заполнение'
        },
        browser_navigate: {
          done: 'Открыл страницу',
          pending: 'Открытие страницы',
          pendingAction: 'Открытие'
        },
        browser_snapshot: {
          done: 'Сделал снимок страницы',
          pending: 'Создание снимка страницы',
          pendingAction: 'Создание'
        },
        browser_take_screenshot: {
          done: 'Сделал скриншот',
          pending: 'Создание скриншота',
          pendingAction: 'Создание'
        },
        browser_type: {
          done: 'Ввёл текст на странице',
          pending: 'Ввод текста на странице',
          pendingAction: 'Ввод'
        },
        clarify: {
          done: 'Задал вопрос',
          pending: 'Задаёт вопрос',
          pendingAction: 'Вопрос'
        },
        cronjob: {
          done: 'Задание cron',
          pending: 'Создание cron-задания',
          pendingAction: 'Создание'
        },
        edit_file: {
          done: 'Изменил файл',
          pending: 'Редактирование файла',
          pendingAction: 'Редактирование'
        },
        execute_code: {
          done: 'Выполнил код',
          pending: 'Выполнение скрипта',
          pendingAction: 'Скрипт'
        },
        image_generate: {
          done: 'Сгенерировал изображение',
          pending: 'Генерация изображения',
          pendingAction: 'Генерация'
        },
        list_files: {
          done: 'Список файлов',
          pending: 'Составление списка файлов',
          pendingAction: 'Список'
        },
        memory: {
          done: 'Сохранил в память',
          pending: 'Сохранение в память',
          pendingAction: 'Сохранение'
        },
        patch: {
          done: 'Пропатчил файл',
          pending: 'Патч файла',
          pendingAction: 'Патч'
        },
        read_file: {
          done: 'Прочитал файл',
          pending: 'Чтение файла',
          pendingAction: 'Чтение'
        },
        search_files: {
          done: 'Нашёл файлы',
          pending: 'Поиск файлов',
          pendingAction: 'Поиск'
        },
        session_search_recall: {
          done: 'Искал в истории сессий',
          pending: 'Поиск в истории сессий',
          pendingAction: 'Поиск'
        },
        terminal: {
          done: 'Выполнил команду',
          pending: 'Выполнение команды',
          pendingAction: 'Выполнение'
        },
        todo: {
          done: 'Обновил задачи',
          pending: 'Обновление задач',
          pendingAction: 'Обновление'
        },
        vision_analyze: {
          done: 'Проанализировал изображение',
          pending: 'Анализ изображения',
          pendingAction: 'Анализ'
        },
        web_extract: {
          done: 'Прочитал страницу',
          pending: 'Чтение страницы',
          pendingAction: 'Чтение'
        },
        web_search: {
          done: 'Искал в веб',
          pending: 'Поиск в веб',
          pendingAction: 'Поиск'
        },
        write_file: {
          done: 'Изменил файл',
          pending: 'Редактирование файла',
          pendingAction: 'Редактирование'
        }
      }
    }
  },
  prompts: {
    gatewayDisconnected: 'Шлюз Hermes не подключён',
    sudoSendFailed: 'Не удалось отправить пароль sudo',
    secretSendFailed: 'Не удалось отправить секрет',
    sudoTitle: 'Пароль администратора',
    sudoDesc:
      'Hermes требуется ваш пароль sudo для выполнения привилегированной команды. Он отправляется только в локальный агент.',
    sudoPlaceholder: 'пароль sudo',
    secretTitle: 'Требуется секрет',
    secretDesc: 'Hermes требуются учётные данные для продолжения.',
    secretPlaceholder: 'значение секрета'
  },
  desktop: {
    audioReadFailed: 'Не удалось прочитать записанное аудио',
    sessionUnavailable: 'Сессия недоступна',
    createSessionFailed: 'Не удалось создать новую сессию',
    promptFailed: 'Сбой промпта',
    providerCredentialRequired: 'Добавьте учётные данные провайдера перед отправкой первого сообщения.',
    emptySlashCommand: 'пустая слэш-команда',
    desktopCommands: 'Команды десктопа',
    skillCommandsAvailable: count =>
      pluralRu(count, {
        one: `Доступна ${count} команда навыка.`,
        few: `Доступны ${count} команды навыков.`,
        many: `Доступно ${count} команд навыков.`
      }),
    warningLine: message => `предупреждение: ${message}`,
    yoloArmed: 'YOLO включён для этого чата',
    yoloOff: 'YOLO выключен',
    yoloSystem: active => `YOLO ${active ? 'вкл' : 'выкл'} для этой сессии`,
    yoloTitle: 'YOLO',
    yoloToggleFailed: 'Не удалось переключить YOLO',
    profileStatus: current =>
      `Профиль: ${current}. Используйте /profile <имя> или «Новую сессию» для старта чата в другом профиле.`,
    unknownProfile: 'Неизвестный профиль',
    noProfileNamed: (target, available) => `Нет профиля «${target}». Доступны: ${available}`,
    newChatsProfile: name => `Новые чаты будут использовать профиль ${name}.`,
    setProfileFailed: 'Не удалось установить профиль',
    sttDisabled: 'Распознавание речи отключено в настройках.',
    stopFailed: 'Не удалось остановить',
    regenerateFailed: 'Не удалось регенерировать',
    editFailed: 'Не удалось изменить',
    resumeFailed: 'Не удалось возобновить',
    resumeStrandedTitle: 'Не удалось загрузить эту сессию',
    resumeStrandedBody:
      'Подключение к этой сессии прервано, и автоматические повторы исчерпаны. Проверьте, что шлюз запущен, затем попробуйте снова.',
    resumeRetry: 'Повторить',
    nothingToBranch: 'Нечего ветвить',
    branchNeedsChat: 'Начните или возобновите чат перед ветвлением.',
    sessionBusy: 'Сессия занята',
    branchStopCurrent: 'Остановите текущий шаг перед ветвлением этого чата.',
    branchNoText: 'У этого сообщения нет текста для ветвления.',
    branchTitle: n => `Черновик: Ветвь #${n}`,
    branchFailed: 'Ветвление не удалось',
    deleteFailed: 'Удаление не удалось',
    archived: 'Архивировано',
    archiveFailed: 'Архивирование не удалось',
    cwdChangeFailed: 'Не удалось сменить рабочий каталог',
    cwdStagedTitle: 'Рабочий каталог подготовлен',
    cwdStagedMessage: 'Перезапустите десктопный бэкенд для применения изменений cwd к этой активной сессии.',
    modelSwitchFailed: 'Сбой переключения модели',
    sessionExported: 'Сессия экспортирована',
    sessionExportFailed: 'Не удалось экспортировать сессию',
    imageSaved: 'Изображение сохранено',
    downloadStarted: 'Загрузка началась',
    restartToUseSaveImage: 'Перезапустите Hermes Desktop для использования «Сохранить изображение».',
    restartToSaveImages: 'Перезапустите Hermes Desktop для сохранения изображений',
    imageDownloadFailed: 'Не удалось скачать изображение',
    openImage: 'Открыть изображение',
    downloadImage: 'Скачать изображение',
    savingImage: 'Сохранение изображения',
    imagePreviewFailed: 'Сбой предпросмотра изображения',
    imageAttach: 'Прикрепление изображения',
    imageWriteFailed: 'Не удалось записать изображение на диск.',
    imageAttachFailed: 'Сбой прикрепления изображения',
    attachImages: 'Прикрепить изображения',
    clipboard: 'Буфер обмена',
    noClipboardImage: 'В буфере обмена нет изображения',
    clipboardPasteFailed: 'Сбой вставки из буфера обмена',
    dropFiles: 'Перетащите файлы',
    handoff: {
      pickPlatform: 'Выберите назначение',
      success: platform => `Передано в ${platform}. Возобновите здесь в любой момент.`,
      systemNote: platform => `↻ Передано в ${platform} — возобновите здесь в любой момент.`,
      failed: error => `Передача не удалась: ${error}`,
      timedOut: 'Таймаут ожидания шлюза. Запущен ли `hermes gateway`?'
    }
  },
  errors: {
    genericFailure: 'Что-то пошло не так',
    boundaryTitle: 'Что-то сломалось в интерфейсе',
    boundaryDesc: 'В представлении возникла ошибка. Ваши чаты и настройки не затронуты.',
    reloadWindow: 'Перезагрузить окно',
    openLogs: 'Открыть логи'
  },
  ui: {
    search: {
      clear: 'Очистить поиск'
    },
    pagination: {
      label: 'Навигация по страницам',
      previous: 'Назад',
      previousAria: 'Перейти к предыдущей странице',
      next: 'Далее',
      nextAria: 'Перейти к следующей странице'
    },
    sidebar: {
      title: 'Боковая панель',
      description: 'Показывает мобильную боковую панель.',
      toggle: 'Скрыть/показать боковую панель'
    }
  }
}
