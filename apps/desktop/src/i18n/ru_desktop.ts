import type { TranslationOverrides } from './define-locale'

const ruNum = (count: number | string) => (typeof count === 'string' ? Number(count) || 0 : count)

const RU_NOUN = (count: number | string, one: string, few: string, many: string) => {
  const c = ruNum(count)
  const n = Math.abs(c) % 10
  const nn = Math.abs(c) % 100
  return n === 1 && nn !== 11 ? one : n >= 2 && n <= 4 && (nn < 12 || nn > 14) ? few : many
}

// The desktop surface's copy, composed by ru.ts.
export const ruDesktop = {
  desktop: {
    audioReadFailed: 'Не удалось прочитать записанное аудио',
    sessionUnavailable: 'Сеанс недоступен',
    createSessionFailed: 'Не удалось создать новый сеанс',
    sendFailedTitle: 'Сообщение не отправлено',
    sendFailedBody:
      'Ваше сообщение вернулось в поле ввода. Отправьте его снова после восстановления соединения.',
    promptFailed: 'Промпт не удался',
    providerCredentialRequired: 'Добавьте учётные данные провайдера перед отправкой первого сообщения.',
    emptySlashCommand: 'пустая слэш-команда',
    desktopCommands: 'Команды desktop',
    skillCommandsAvailable: count =>
      `${count} ${RU_NOUN(count, 'команда навыка', 'команды навыка', 'команд навыка')} доступно.`,
    warningLine: message => `предупреждение: ${message}`,
    yoloArmed: 'YOLO включён для этого чата',
    yoloOff: 'YOLO выключен',
    yoloSystem: active => `YOLO ${active ? 'включён' : 'выключен'} для этого сеанса`,
    yoloTitle: 'YOLO',
    yoloToggleFailed: 'Не удалось переключить YOLO',
    profileStatus: current =>
      `Профиль: ${current}. Используйте /profile <name> или выборку «Новый сеанс», чтобы начать чат в другом профиле.`,
    unknownProfile: 'Неизвестный профиль',
    noProfileNamed: (target, available) => `Нет профиля «${target}». Доступны: ${available}`,
    newChatsProfile: name => `Новые чаты будут использовать профиль ${name}.`,
    setProfileFailed: 'Не удалось установить профиль',
    sttDisabled: 'Распознавание речи отключено в настройках.',
    stopFailed: 'Остановка не удалась',
    regenerateFailed: 'Повторная генерация не удалась',
    editFailed: 'Изменение не удалось',
    editTurnUnavailable: 'Этого хода больше нет в истории сервера (возможно, он был сжат).',
    resumeFailed: 'Возобновление не удалось',
    resumeStrandedTitle: 'Не удалось загрузить этот сеанс',
    resumeStrandedBody:
      'Соединение с этим сеансом оборвалось, и автоматические повторные попытки исчерпаны. Проверьте, что шлюз работает, и попробуйте снова.',
    poolSlotTimeoutBody:
      'Все слоты локальных бэкендов профилей заняты. Увеличьте Warm Bot Backends в «Настройки» → «Дополнительно» или повторите попытку после освобождения неактивного бэкенда.',
    poolSlotTimeoutOpenSettings: 'Открыть расширенные настройки',
    resumeRetry: 'Повторить',
    nothingToBranch: 'Нечего ветвить',
    branchNeedsChat: 'Начните или возобновите чат перед ветвлением.',
    sessionBusy: 'Сеанс занят',
    branchStopCurrent: 'Остановите текущий ход перед ветвлением этого чата.',
    branchNoText: 'В этом сообщении нет текста, от которого можно ответвить.',
    branchTitle: n => `Черновик: Ветка #${n}`,
    branchFailed: 'Ветвление не удалось',
    deleteFailed: 'Удаление не удалось',
    archived: 'В архиве',
    archiveFailed: 'Архивирование не удалось',
    restored: 'Восстановлено',
    unarchiveFailed: 'Не удалось восстановить',
    cwdChangeFailed: 'Изменение рабочего каталога не удалось',
    cwdStagedTitle: 'Рабочий каталог поставлен в очередь',
    cwdStagedMessage: 'Перезапустите бэкенд desktop, чтобы применить изменения cwd к этому активному сеансу.',
    modelSwitchConfirmBody: 'Эта смена модели требует подтверждения.',
    modelSwitchConfirmLabel: 'Всё равно переключить',
    modelSwitchConfirmTitle: (model: string) => `Переключиться на ${model}?`,
    modelSwitchConfirmTitleFallback: 'Сменить модель?',
    modelSwitchFailed: 'Смена модели не удалась',
    modelSwitchKeepLabel: 'Оставить текущую модель',
    modelSwitchStaleNotice: 'Выбор изменился — смена модели не применена.',
    sessionExported: 'Сеанс экспортирован',
    sessionExportFailed: 'Не удалось экспортировать сеанс',
    imageSaved: 'Изображение сохранено',
    downloadStarted: 'Загрузка началась',
    restartToUseSaveImage: 'Перезапустите Hermes Desktop, чтобы использовать «Сохранить изображение».',
    restartToSaveImages: 'Перезапустите Hermes Desktop, чтобы сохранять изображения',
    imageDownloadFailed: 'Загрузка изображения не удалась',
    openImage: 'Открыть изображение',
    downloadImage: 'Скачать изображение',
    savingImage: 'Сохранение изображения',
    imagePreviewFailed: 'Предпросмотр изображения не удался',
    imageAttach: 'Прикрепление изображения',
    imageWriteFailed: 'Не удалось записать изображение на диск.',
    imageAttachFailed: 'Прикрепление изображения не удалось',
    attachImages: 'Прикрепить изображения',
    clipboard: 'Буфер обмена',
    noClipboardImage: 'Изображение в буфере обмена не найдено',
    clipboardPasteFailed: 'Вставка из буфера обмена не удалась',
    dropFiles: 'Перетащите файлы',
    handoff: {
      pickPlatform: 'Выберите назначение',
      success: platform => `Передаём в ${platform}. Возобновите здесь в любой момент.`,
      systemNote: platform => `↻ Передано в ${platform} — возобновите здесь в любой момент.`,
      failed: error => `Передача не удалась: ${error}`,
      timedOut: 'Превышено время ожидания шлюза. Выполняется ли `hermes gateway`?'
    }
  }
} satisfies Pick<TranslationOverrides, 'desktop'>
