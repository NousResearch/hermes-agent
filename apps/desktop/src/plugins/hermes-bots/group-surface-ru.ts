import { GROUP_CHAT_LOCALES } from './group-chat-locales'
import type { BotsMessages } from './i18n-messages'
import { ROSTER_CHROME } from './roster-chrome-locales'

export const GROUP_SURFACE_RU = {
  group: {
    ...GROUP_CHAT_LOCALES.ru,
    newTitle: 'Новый групповой чат',
    manageDesc:
      'Агент может участвовать в нескольких групповых чатах. Состав групп синхронизируется на всех компьютерах.',
    manageTitle: 'Управление группами',
    settingsTitle: 'Настройки группы',
    settingsDesc: 'Измените название или изображение группы. Участники и история сохранятся.',
    nameLabel: 'Название группы',
    holdDetection: 'Распознавать команды остановки',
    holdDetectionHint:
      'Команды остановки в групповом чате приостанавливают указанных участников до следующего упоминания.',
    compressHistory: 'Сжать историю',
    compressHistoryHint: member => `Сжать скрытую историю группового чата для ${member}, чтобы избежать пустых ответов`,
    compressing: member => `Сжатие истории группового чата для ${member}…`,
    compressDone: (member, compressed, detail) =>
      `Сжато сессий группового чата для ${member}: ${compressed}${detail ? ` — ${detail}` : ''}`,
    compressNothing: member => `Нечего сжимать для ${member} — сессий группового чата пока нет`,
    compressFailed: (member, error) => `Не удалось сжать историю группового чата для ${member}: ${error}`,
    searchToAdd: 'Поиск агентов для добавления',
    searchToAddPlaceholder: 'Найти агентов для добавления…',
    removeFromSelection: 'Убрать из выбранных',
    disbandTitle: 'Распустить групповой чат?',
    deleteTitle: 'Удалить групповой чат?',
    deleteAction: 'Удалить',
    composerPlaceholder: 'Напишите что-нибудь — все агенты в этом групповом чате увидят сообщение.',
    slashCommandsUnsupported:
      'Команды с косой чертой недоступны в групповых чатах. Используйте их в личном чате с агентом.',
    attachHint: 'Прикрепить файлы — все отвечающие агенты их увидят',
    newThread: 'Новая ветка',
    reply: 'Ответить',
    replyInThread: 'Ответить в ветке',
    replyInThreadPlaceholder: 'Ответить в ветке…',
    openThread: 'Открыть эту ветку',
    collapseThread: 'Свернуть ветку',
    collapseThreadLabel: 'Свернуть эту ветку',
    activity: 'Активность',
    noActivityYet: 'В этом ходе пока нет активности.',
    showActivity: 'Показать активность группового чата',
    hideActivity: 'Скрыть активность группового чата',
    stop: 'Остановить',
    stopHint: 'Остановить выполнение — прервать активного участника и приостановить остальных',
    heldMembersStatus: members => `Приостановлены: ${members}`,
    holdReleaseHint: 'Упомяните приостановленного агента или отправьте @all resume для продолжения.',
    needsYourInput: 'Агенту в этом групповом чате нужен ваш ответ',
    noMembersToSend: group =>
      `В группе «${group}» нет получателей — добавьте агента или откройте групповой чат заново, если участники ещё загружаются.`,
    pictureGenerationFailed: 'Не удалось создать изображение группы',
    nameTaken: name => `Группа «${name}» уже существует.`,
    you: 'Вы',
    settingsHint: group => `Настройки группы — изменить название или изображение «${group}»`,
    settingsLabel: group => `Настройки группы «${group}»`,
    disbandHint: group => `Распустить групповой чат «${group}»`,
    disbandLabel: group => `Распустить «${group}»`,
    disbandAction: 'Распустить',
    disbanding: 'Роспуск…',
    disbandDone: 'Группа распущена',
    disbanded: group => `Группа «${group}» распущена`,
    stopped: group => `Группа «${group}» остановлена — оставшиеся ходы приостановлены до продолжения`,
    removeAttachment: 'Удалить вложение',
    threadFallback: 'Ветка',
    replyCount: replies =>
      `${replies} ${replies % 10 === 1 && replies % 100 !== 11 ? 'ответ' : replies % 10 >= 2 && replies % 10 <= 4 && (replies % 100 < 12 || replies % 100 > 14) ? 'ответа' : 'ответов'}`,
    dropToThread: 'Перетащите, чтобы прикрепить к ответу в ветке',
    dropToRoom: 'Перетащите для прикрепления — все отвечающие агенты увидят файл',
    waitingForAnswer: 'Ожидание вашего ответа…',
    memberThinking: name => `${name} думает…`,
    roomWorking: 'Групповой чат работает…',
    messageRoom: group => `Написать в «${group}»`,
    newThreadPlaceholder: group => `Новая ветка в «${group}»… (@имя для участника, @everyone для всех)`,
    everyoneMeta: 'Все агенты в групповом чате',
    commandApproval: 'подтверждение команды',
    answerFailed: (handle, error) => `Не удалось отправить ответ @${handle}: ${error}`,
    wantsToRunCommand: handle => `@${handle} хочет выполнить команду:`,
    asks: handle => `@${handle} спрашивает:`,
    answerTo: member => `Ответить @${member}`
  },
  roster: {
    ...ROSTER_CHROME.ru,
    search: 'Поиск агентов и групповых чатов',
    searchPlaceholder: 'Поиск агентов и групповых чатов…',
    newBotOrGroup: 'Новый агент или групповой чат',
    groupChats: 'Групповые чаты',
    emptyTitle: 'Агентов пока нет',
    emptyDesc: 'Создайте первого агента.',
    noMatchQuery: query => `Нет агентов или групповых чатов по запросу «${query}»`,
    noMatchQueryOn: (query, gateway) => `Нет агентов или групповых чатов по запросу «${query}» на ${gateway}`,
    noMatchFiltersOn: gateway => `Нет агентов или групповых чатов с такими фильтрами на ${gateway}`,
    noMatchFilters: 'Нет агентов или групповых чатов с такими фильтрами.',
    clearFilters: 'Сбросить фильтры',
    allHidden: 'Все агенты скрыты',
    allHiddenDesc: 'Они продолжают работать и сохраняют историю.',
    showHidden: 'Показать скрытых агентов',
    noHiddenMatch: 'Нет скрытых агентов с такими фильтрами.',
    hiddenFromRoster: 'Скрыт из списка',
    pinned: 'Закреплён',
    needsAttention: 'требует внимания',
    needsInput: 'Нужен ваш ответ',
    botsAndGroups: 'Агенты и групповые чаты',
    botsOnly: 'Только агенты',
    groupsOnly: 'Только групповые чаты',
    anyActivity: 'Любая активность',
    activeNow: 'Активны сейчас',
    recentlyActive: 'Недавно активные',
    older: 'Более ранние',
    gatewayRemoved: 'Компьютер удалён',
    onDemand: 'По запросу',
    ready: 'Готов',
    statusUnknown: 'Статус неизвестен',
    unavailable: 'Недоступен',
    retryNow: 'Повторить сейчас',
    rosterUnavailable: reason =>
      `Список недоступен: ${reason}. Если ваша версия Hermes не поддерживает profiles.list, обновите и перезапустите Hermes.`,
    waitingForGateway:
      'Ожидание подключения к компьютеру… (удалённые компьютеры могут отвечать несколько секунд; повторная попытка автоматически)'
  },
  sections: {
    newSection: 'Новый раздел',
    newTitle: 'Новый раздел',
    renameTitle: 'Переименовать раздел',
    nameLabel: 'Название раздела',
    namePlaceholder: 'Например, Клиенты',
    create: 'Создать',
    rename: 'Переименовать…',
    moveUp: 'Выше',
    moveDown: 'Ниже',
    unassigned: 'Без раздела',
    options: name => `Параметры раздела «${name}»`,
    headingTip: 'Перетащите сюда агентов · двойной щелчок для переименования',
    emptyHint: 'Перетащите сюда агентов',
    moveTo: 'Переместить в раздел',
    newSectionEllipsis: 'Новый раздел…',
    removeFromSection: 'Убрать из раздела',
    deleted: (name, count) =>
      count === 0 ? `Раздел «${name}» удалён` : `Раздел «${name}» удалён — агентов без раздела: ${count}`,
    undo: 'Отменить'
  },
  bot: {
    newChatWith: 'Новый чат с этим агентом',
    newTitle: 'Новый агент',
    thisDevice: 'Это устройство',
    removeFromAllGroups: 'Удалить из всех групп',
    openBotChat: 'Открыть чат с агентом',
    openBotChatInNewTab: 'Открыть чат с агентом в новой вкладке',
    openingChat: 'Открытие чата…',
    pinToTop: 'Закрепить сверху',
    unpin: 'Открепить',
    pinnedToast: name => `${name} закреплён сверху`,
    unpinnedToast: name => `${name} откреплён`,
    hide: 'Скрыть',
    unhide: 'Показать',
    hiddenToast: name => `${name} скрыт — нажмите значок глаза над списком агентов, чтобы его увидеть`,
    unhiddenToast: name => `${name} снова в списке`,
    groupsMenu: groups => `Группы: ${groups}…`,
    manageGroups: 'Управление группами…',
    metadataLoadFailed: 'Не удалось загрузить метаданные агента',
    loadFailed: 'Не удалось загрузить агента',
    groupsLoadFailed: 'Не удалось загрузить группы агента',
    attentionFallback: 'Требует внимания',
    attentionProviderAuth: 'Войдите заново для этого агента',
    attentionQuota: 'Квота или баланс исчерпаны',
    attentionMissingConfig: 'Провайдер не настроен — выполните hermes model',
    attentionBlocked: 'Агент заблокирован — см. последнее сообщение',
    editMenu: 'Изменить…',
    duplicate: 'Дублировать',
    duplicateFailed: 'Не удалось создать копию'
  },
  avatar: {
    upload: 'Загрузить',
    generate: 'Создать',
    generating: 'Создание…'
  },
  screen: {
    autoOpenMenu: 'Открывать экран, когда агент его использует',
    autoOpenOnToast: name => `Экран ${name} открывается, когда агент начинает использовать рабочий стол`,
    autoOpenOffToast: name => `Экран ${name} остаётся закрытым, пока вы его не откроете`,
    menu: 'Открыть экран'
  }
} satisfies Pick<BotsMessages, 'group' | 'roster' | 'sections'> & {
  bot: Pick<
    BotsMessages['bot'],
    | 'newChatWith'
    | 'newTitle'
    | 'thisDevice'
    | 'removeFromAllGroups'
    | 'openBotChat'
    | 'openBotChatInNewTab'
    | 'openingChat'
    | 'pinToTop'
    | 'unpin'
    | 'pinnedToast'
    | 'unpinnedToast'
    | 'hide'
    | 'unhide'
    | 'hiddenToast'
    | 'unhiddenToast'
    | 'groupsMenu'
    | 'manageGroups'
    | 'metadataLoadFailed'
    | 'loadFailed'
    | 'groupsLoadFailed'
    | 'attentionFallback'
    | 'attentionProviderAuth'
    | 'attentionQuota'
    | 'attentionMissingConfig'
    | 'attentionBlocked'
    | 'editMenu'
    | 'duplicate'
    | 'duplicateFailed'
  >
  avatar: Pick<BotsMessages['avatar'], 'upload' | 'generate' | 'generating'>
  screen: Pick<BotsMessages['screen'], 'menu' | 'autoOpenMenu' | 'autoOpenOnToast' | 'autoOpenOffToast'>
}
