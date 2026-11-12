/**
 * The complete Russian Bot Mode bundle. The group surface (group chat, roster,
 * sections and the shared bot/avatar/screen keys) already lives in
 * `./group-surface-ru.ts` and the shared locale tables; this file reuses them
 * per section and adds the rest of the Bot Mode copy, so `ru` never falls back
 * to the English bundle. Nouns match core `ru.ts`: бот, профиль, шлюз, навык,
 * набор инструментов, сеанс, задача (cron).
 */

import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { CROSS_GATEWAY_GROUP_LOCALES } from './cross-gateway-locales'
import { GROUP_SURFACE_RU } from './group-surface-ru'
import { type BotsMessages } from './i18n-messages'

/** One (1, 21), few (2–4, 22–24) or many (0, 5–20, 25…) form of a noun after a count. */
function russianNoun(count: number, one: string, few: string, many: string): string {
  const n = Math.abs(count) % 10
  const nn = Math.abs(count) % 100

  return n === 1 && nn !== 11 ? one : n >= 2 && n <= 4 && (nn < 12 || nn > 14) ? few : many
}

/** Core's weekday labels (`cron.days.*`) after «каждый», which agrees with the day's gender. */
const EVERY_WEEKDAY: Record<string, string> = {
  понедельник: 'каждый понедельник',
  вторник: 'каждый вторник',
  среда: 'каждую среду',
  четверг: 'каждый четверг',
  пятница: 'каждую пятницу',
  суббота: 'каждую субботу',
  воскресенье: 'каждое воскресенье'
}

function everyWeekday(day: string): string {
  return EVERY_WEEKDAY[String(day).toLowerCase()] ?? `еженедельно (${day})`
}

export const BOTS_RU: BotsMessages = {
  canonical: CANONICAL_GROUP_LOCALES.ru,
  crossGateway: CROSS_GATEWAY_GROUP_LOCALES.ru,
  editor: {
    fullConfigHint: 'Для полной настройки нужен более новый шлюз (перезапустите его после обновления Hermes).',
    liveCapabilities: 'Возможности (применяются сразу — навыки, инструменты, MCP)',
    editSoul: 'SOUL.md (персона и протокол обмена сообщениями между агентами)',
    remoteCapabilitiesHint:
      'Для удалённых возможностей нужна более новая версия Hermes Desktop. Изменения модели и SOUL применятся после сохранения.',
    skillsEnabled: (enabled, total) => `Навыки (включено: ${enabled}/${total})`,
    toolsetsEnabled: (enabled, total) =>
      `Наборы инструментов (включено: ${enabled}/${total} — если снять все отметки, вернётся набор по умолчанию)`,
    mcpServers: 'MCP-серверы',
    providerCustom: 'Провайдер (свой)',
    modelCustom: 'Модель (своя)',
    backToDropdowns: '← Вернуться к спискам',
    inheritLaunch: 'Наследовать (профиль запуска)',
    enterManually: '✏️ Ввести вручную…',
    gatewayDefault: 'по умолчанию для шлюза',
    modelNameExample: 'например, название модели',
    modelSwitchFailed: 'Не удалось сменить модель',
    newDescription: 'Напарник со своим именем, памятью, навыками и чатом. Может писать другим вашим агентам.',
    name: 'Имя',
    title: 'Отображаемое имя',
    description: 'Описание',
    createOn: 'Где создать',
    general: 'Общие',
    capabilities: 'Возможности',
    skills: 'Навыки',
    tools: 'Инструменты',
    cloneFrom: 'Клонировать из профиля',
    freshProfile: 'Новый профиль (встроенные навыки)',
    inheritedModel: 'наследуется от профиля запуска',
    soul: 'SOUL.md (необязательно — заменяет сгенерированную персону)',
    shareKeys: 'Использовать общие ключи и аккаунты с основным профилем',
    shareKeysOn: target => `Использовать общие ключи и аккаунты с профилем по умолчанию на ${target}`,
    shareKeysHint:
      'Подписки, входы через OAuth и API-ключи остаются общими с профилем по умолчанию (а не копируются), поэтому обновление токена в одном профиле не сбрасывает его в другом. Снимите флажок, чтобы получить отдельную копию-снимок.',
    needsModel:
      'Для него пока не готов ни один провайдер модели, поэтому бот не представился. Выберите провайдера и модель в разделе «Дополнительно».',
    configureModel: 'Настроить модель',
    createEmpty: 'Создать пустым (без встроенных навыков)',
    nameTakenHint: 'Это имя занято — выберите другое, прежде чем настраивать возможности.',
    nameFirstHint:
      'Сначала дайте боту имя — при открытии этой вкладки создаётся черновик профиля (он удалится, если вы отмените создание).',
    newerDesktop: 'Для навыков нужна более новая версия Hermes Desktop.',
    newerGateway: 'Для каталога возможностей нужен более новый шлюз (перезапустите его после обновления Hermes).',
    emptySkillsHint: 'Отмечено «Создать пустым» — встроенные навыки не будут установлены.',
    defaultToolsHint: 'Если отметить все наборы инструментов (или ни одного), они будут работать по умолчанию.',
    catalog: 'каталог',
    catalogInstalled: 'каталог · установлен',
    mcpHint:
      'Настроенные серверы копируются из основного профиля; записи каталога — это встроенное меню MCP. Записи, которым нужны API-ключи, сначала проходят настройку (учётные данные зависят от настройки общих ключей).',
    creating: 'Создание…',
    createBot: 'Создать бота',
    auto: 'Авто',
    autoHint: 'Авто — выбирается по имени',
    unlock: 'Разблокировать',
    lockFace: 'Зафиксировать лицо',
    lockedHint: 'Лицо зафиксировано — переименование его не изменит.',
    unlockedHint: 'Лицо меняется вместе с именем.',
    noImageModel:
      'Нет доступной модели изображений. Если вы только что её включили (или обновили Hermes), перезапустите шлюз: Ctrl+K → «Перезапустить шлюз».',
    checkingImage: 'Проверка бэкенда изображений…',
    chooseImage: 'Выбрать изображение…',
    editDescription: (name, profile) => `Внешний вид и роль бота ${name} (${profile}).`,
    nameTaken: name => `Агент с именем «${name}» уже существует.`,
    nameTakenOn: (name, target) => `Агент с именем «${name}» уже есть на ${target}.`,
    currentConnection: name => `${name} (текущее)`,
    remoteHint: target =>
      `Агент будет создан на ${target} и появится в списке как бот другого подключения. Чат будет идти через тот компьютер.`,
    cloneFromOn: target => `Клонировать из профиля (на ${target})`,
    catalogHint: source => `Каталог из ${source} — неотмеченные навыки будут отключены после создания.`,
    sectionsFailed: sections => `Не удалось применить некоторые разделы: ${sections}`,
    updated: name => `${name} обновлён`,
    created: name => `Бот «${name}» создан`,
    createdOn: (name, target) => `Бот «${name}» создан на ${target}`
  },
  roster: GROUP_SURFACE_RU.roster,
  sections: GROUP_SURFACE_RU.sections,
  bot: {
    ...GROUP_SURFACE_RU.bot,
    editTitle: 'Изменить профиль',
    helpPromptPlaceholder: 'С чем должен помогать этот бот?',
    descriptionHint: 'Оставьте пустым, чтобы сгенерировать по имени и описанию бота.',
    deleteTitle: 'Удалить бота и профиль?',
    createFirstHint: 'Откройте панель «Боты» и нажмите «Новый бот».',
    createFailed: 'Пока не удалось создать профиль',
    advanced: 'Дополнительно',
    advancedHint: 'Дополнительно — модель, навыки, наборы инструментов, SOUL.md',
    advancedFailed: 'Не удалось применить дополнительные настройки',
    openAnotherChatUnsupported: 'Обновите Hermes Desktop, чтобы открыть ещё один чат с ботом.',
    remoteConnectionsUnsupported: 'Обновите Hermes Desktop, чтобы общаться с ботами на других подключениях.',
    openNeedsUpdateTitle: 'Этот бот работает на старой версии Hermes',
    openNeedsUpdateMessage: connectionLabel => `Обновите ${connectionLabel} и повторите попытку.`,
    openUnreachableTitle: 'Hermes не удалось связаться с компьютером, на котором работает этот бот',
    openUnreachableMessage: 'Убедитесь, что он в сети, и повторите попытку.',
    openChatFailedTitle: botName => `Не удалось открыть чат с ${botName}`,
    openChatFailedMessage: 'Повторите попытку.',
    openGateways: 'Открыть раздел «Шлюзы»',
    chatEmpty: 'Напишите что-нибудь, чтобы начать.',
    kickoff: 'Привет, расскажи о себе!'
  },
  avatar: {
    ...GROUP_SURFACE_RU.avatar,
    classicShapes: 'Классические фигуры',
    blobFromName: 'Лицо-клякса — рисуется по имени бота',
    unlockFollowsName: 'Разблокировать — лицо снова будет меняться вместе с именем бота',
    randomize: 'Случайный вариант',
    tabBot: 'Бот',
    tabGenerate: 'Создать',
    tabPet: 'Питомец',
    removeImage: 'Удалить изображение — использовать фигуру',
    removeBackToShape: 'Удалить — вернуть аватар-фигуру',
    describePlaceholder: 'Опишите аватар…',
    describeHint:
      'Оставьте пустым, чтобы сгенерировать автоматически из имени, отображаемого имени, описания и списка агентов для обмена сообщениями.',
    matchTheName: 'Подобрать по имени',
    pickPet: 'Выберите питомца для изображения профиля этого бота.',
    petLoadFailed: 'Не удалось загрузить этого питомца — попробуйте другого.',
    imageTooLarge: 'Изображение слишком большое (максимум 15 МБ).',
    generationFailed: 'Не удалось создать аватар',
    savedLocally: 'Внешний вид сохранён локально, но сохранить его удалённо не удалось',
    savedLocallyDescriptionFailed: 'Внешний вид сохранён локально, но обновить описание не удалось'
  },
  group: GROUP_SURFACE_RU.group,
  tools: {
    installHint: name => `Установить «${name}» и добавить в список выше`,
    installed: name => `Навык «${name}» установлен`,
    installFailed: name => `Не удалось установить «${name}»`,
    searchHint: 'Поиск в сообществе и известных источниках — может занять около 10 секунд…',
    resizeHint: 'Потяните за угол, чтобы изменить размер.',
    addServerFailed: 'Не удалось добавить сервер',
    noTarget: 'Нет целевого профиля',
    setKeyFailed: key => `Не удалось задать ${key}`,
    configured: name => `${name} настроен`,
    authenticated: name => `${name}: вход выполнен`,
    testFailed: 'Проверка сервера после настройки не прошла',
    completeSignIn: 'Завершите вход в браузере…',
    needsSetup: keys => `нужна настройка (${keys}) — перезапустите шлюз, чтобы настроить прямо в приложении`,
    setUpDone: 'настроен ✓',
    saveTest: 'Сохранить и проверить',
    authorizing: 'Авторизация…',
    working: 'Выполняется…',
    setupFailed: 'Настройка не удалась',
    signIn: 'Войти…',
    setUp: 'Настроить…',
    skillsHub: 'Хаб навыков Hermes',
    filterSkills: 'Фильтр навыков…',
    searchHub: 'Поиск в хабе (сообщество и известные источники)…',
    noMcpServers: 'Нет MCP-серверов: ни настроенных, ни в каталоге.'
  },
  screen: {
    ...GROUP_SURFACE_RU.screen,
    title: 'Экран',
    unsupportedTitle: 'На этом хосте нет экрана бота',
    unsupportedBody: 'Экраны ботов работают на хостах шлюза с Linux. Этот бот использует собственный дисплей хоста.',
    notInstalledTitle: 'Не установлены пакеты экрана',
    notInstalledBody:
      'Чтобы у этого бота появился экран, на хосте шлюза нужны TigerVNC и базовые компоненты Xfce. Выполните на хосте:',
    installHint:
      'Выполняется на хосте шлюза от имени пользователя, под которым работает Hermes; пароль sudo запрашивается один раз через Hermes.',
    install: 'Установить на хост',
    installing: 'Установка…',
    installCancelled: 'Установка отменена: пароль sudo не введён.',
    installFailed: 'Установка не удалась. Посмотрите журнал выше или выполните команду на хосте самостоятельно.',
    noPackageManager: 'На хосте шлюза не найден поддерживаемый менеджер пакетов (apt, dnf, pacman).',
    portalTitle: 'Экран',
    portalOpen: 'Открыть',
    heroStopped: 'Экран выключен',
    heroNotInstalled: 'Не установлен на этом хосте',
    heroConnecting: 'Проверка экрана…',
    heroStale: 'Последний кадр — экран недоступен',
    heroSuppressed: 'Скрыт, пока кто-то управляет',
    heroOpenLive: 'Смотреть вживую',
    heroInstall: 'Установить',
    heroStart: 'Запустить',
    portalWatching: 'В эфире · управляет бот',
    portalYouControl: 'В эфире · управляете вы',
    portalOtherControls: 'В эфире · управляет другой зритель',
    portalStopped: 'Остановлен',
    portalNotInstalled: 'Не установлен на хосте',
    portalUnsupported: 'Недоступен на этом хосте',
    portalUnavailable: 'Обновите Hermes бота, чтобы пользоваться экраном',
    portalUnavailableManaged: 'В этом управляемом выпуске Hermes экран пока недоступен',
    managedNotIncludedTitle: 'У этого бота нет экрана',
    managedNotIncludedBody:
      'В Hermes Cloud экран есть у ботов с памятью от 8 ГБ. На бота с меньшим объёмом памяти его установить нельзя.',
    unavailableTitle: 'Для экрана нужна более новая версия Hermes',
    stoppedTitle: 'Экран выключен',
    stoppedBody:
      'Запустите рабочий стол этого бота, чтобы видеть, что он делает, и брать управление на себя, когда вы ему нужны.',
    placementSandbox: backend => `Экран работает в песочнице ${backend} вместе с терминалом`,
    imageSwitchTitle: 'Доступен новый образ песочницы',
    imageSwitchBody: (current, target) =>
      `Ваша песочница всё ещё использует образ ${current} без рабочего стола. Переход на ${target} пересоздаст контейнер, когда бот в следующий раз воспользуется терминалом: файлы в /root и /workspace останутся на этом компьютере, а пакеты, установленные внутри контейнера, будут переустановлены по мере необходимости.`,
    imageSwitchApprove: 'Сменить образ',
    imageSwitchKeep: 'Оставить текущий образ',
    start: 'Запустить экран',
    attaching: 'Подключение к экрану…',
    streamLost: 'Трансляция экрана прервалась',
    reconnect: 'Переподключиться',
    takeOver: 'Взять управление',
    handBack: 'Вернуть управление',
    handBackForce: 'Вернуть управление (принудительно)',
    handBackForceHint:
      'Освободить управление, которое удерживает уже ушедший зритель, например после перезагрузки страницы.',
    openNeedsUpdate: 'Обновите Hermes Desktop, чтобы открывать экраны ботов.',
    youControl: 'Управляете вы',
    otherControls: 'Управляет другой зритель',
    agentControls: 'Управляет бот',
    controlTaken: 'Управление взял другой зритель. Только просмотр.',
    needsHttps:
      'Для экранов ботов нужно защищённое соединение. Чтобы посмотреть этот экран, откройте веб-приложение по HTTPS или через SSH-туннель к 127.0.0.1.'
  },
  cron: {
    untitled: 'Задача без названия',
    nameNul: 'Название задачи не может содержать NUL (U+0000).',
    instructionNul: 'Инструкция задачи не может содержать NUL (U+0000).',
    minutesFromNow: 'мин от текущего момента',
    hoursFromNow: 'ч от текущего момента',
    daysFromNow: 'дн. от текущего момента',
    stopAfter: 'Лимит запусков',
    runsHint: '(пусто — без ограничений)',
    detailDescription: 'Что выполняет эта задача и когда она запустится в следующий раз.',
    status: 'Статус',
    active: 'Активна',
    paused: 'Приостановлена',
    schedule: 'Расписание',
    rawSchedule: 'Расписание (исходное)',
    repeat: 'Повтор',
    nextRun: 'Следующий запуск',
    overdueSince: 'Просрочена с',
    lastRun: 'Последний запуск',
    lastResult: 'Последний результат',
    workdir: 'Рабочий каталог',
    succeeded: 'Успешно',
    failed: 'Ошибка',
    deliveryFailed: 'Выполнена, но доставить результат не удалось',
    blockedConfig: 'Заблокирована настройками (не запускалась)',
    legacyUnsafe:
      'Приостановлена из соображений безопасности: удалите эту устаревшую задачу и создайте её заново, прежде чем запускать.',
    filterHint:
      'В этом профиле есть запланированные задачи, но ни одна не помечена для этого бота. Назовите задачу «[bot:<имя>] …», чтобы она появилась здесь, или найдите их в разделе Cron ниже.',
    needsRosterFirst: 'Сначала этот бот должен появиться в списке.',
    staleNotice: 'Не удалось обновить запланированные задачи. Показан последний полученный список.',
    readFailure: 'Список, скорее всего, на месте — это ошибка чтения, а не удаление.',
    createDesc: bot =>
      `Повторяющаяся задача, которую ${bot} выполняет по расписанию. Запуски сохраняются в истории его чата.`,
    instruction: 'Инструкция',
    whenToRun: 'Когда запускать',
    dayOfMonth: 'День месяца',
    sendResultsTo: 'Куда отправлять результаты',
    runHistoryOnly: 'Только в историю запусков',
    botChatTarget: bot => `Чат с ${bot} (бот ответит)`,
    continuity:
      'Непрерывность: каждый запуск видит результат предыдущего (чтобы не повторяться и продолжать с того же места)',
    onceIn: when => `Однократно (${when})`,
    everyNDays: days => (days === 1 ? 'Каждый день' : `Каждые ${days} ${russianNoun(days, 'день', 'дня', 'дней')}`),
    everyNHours: hours =>
      hours === 1 ? 'Каждый час' : `Каждые ${hours} ${russianNoun(hours, 'час', 'часа', 'часов')}`,
    everyNMinutes: minutes =>
      minutes === 1 ? 'Каждую минуту' : `Каждые ${minutes} ${russianNoun(minutes, 'минуту', 'минуты', 'минут')}`,
    freqOnce: 'Однократно, через…',
    freqHourly: 'Каждый час',
    freqDaily: 'Каждый день',
    freqWeekdays: 'По будням',
    freqWeekly: 'Каждую неделю',
    freqMonthly: 'Каждый месяц',
    freqInterval: 'С интервалом',
    freqAdvanced: 'Своё расписание…',
    unitMinutes: 'мин',
    unitHours: 'ч',
    unitDays: 'дн.',
    runsOnce: (count, unit) => `Выполнится один раз через ${count} ${unit}`,
    runsHourly: 'Выполняется в начале каждого часа',
    runsDaily: time => `Выполняется каждый день в ${time}`,
    runsWeekdays: time => `Выполняется с понедельника по пятницу в ${time}`,
    runsWeekly: (day, time) => `Выполняется ${everyWeekday(day)} в ${time}`,
    runsMonthly: (day, time) => `Выполняется ${day}-го числа каждого месяца в ${time}`,
    runsInterval: (count, unit) => `Выполняется раз в ${count} ${unit}`,
    runsRaw: 'Своё расписание — every Nm/Nh/Nd или cron-выражение из 5 полей',
    timesTotal: count => `, всего ${count} ${russianNoun(count, 'раз', 'раза', 'раз')}`
  }
}
