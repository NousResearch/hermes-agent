/**
 * Russian kanban bundle, registered as `ru` in `KANBAN_LOCALES` (i18n.ts). Typed as the full
 * `KanbanMessages`, so tsc fails on a missing or extra key. Terms follow the core Russian catalog
 * (src/i18n/ru*.ts): воркер, профиль, навык, вспомогательная модель, глубина рассуждений, ток.
 * Column labels are quoted with «» wherever the prose names a column.
 */

import type { KanbanMessages } from './i18n'

/** Russian plural form: 1, 21 → one; 2–4, 22–24 → few; 0, 5–20, 25… → many. */
const plural = (n: number, one: string, few: string, many: string) => {
  const mod10 = n % 10
  const mod100 = n % 100

  if (mod10 === 1 && mod100 !== 11) {
    return one
  }

  return mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14) ? few : many
}

export const ru: KanbanMessages = {
  nav: 'Канбан',
  openBoard: 'Канбан: открыть доску',
  newTaskCommand: 'Канбан: новая задача',
  countTip: (running, ready) => `Канбан — выполняется: ${running}, готово к запуску: ${ready}`,
  col: {
    triage: { label: 'Уточнение', help: 'Сырая идея — агент уточнит её и допишет спецификацию.' },
    todo: { label: 'К выполнению', help: 'Ждёт зависимостей или исполнителя.' },
    scheduled: { label: 'Запланировано', help: 'Ждёт запланированного времени.' },
    ready: {
      label: 'Готово к запуску',
      help: 'Зависимости выполнены — назначьте профиль, и диспетчер запустит задачу.'
    },
    running: { label: 'Выполняется', help: 'Задачу взял воркер — над ней работает агент. Статус ставит диспетчер.' },
    blocked: { label: 'Заблокировано', help: 'Воркер ждёт ответа от человека.' },
    review: { label: 'Проверка', help: 'Агент-рецензент проверяет работу. Статус ставит диспетчер.' },
    done: { label: 'Завершено', help: 'Задача выполнена; зависимые подзадачи переходят в «Готово к запуску».' },
    archived: { label: 'Архив', help: 'Не показывается на доске по умолчанию.' }
  },
  locked: {
    review: 'В колонку «Проверка» карточку переводит диспетчер, когда её берёт агент-рецензент.',
    running: 'В колонку «Выполняется» карточку переводит диспетчер, когда её берёт воркер.',
    scheduled: 'Для колонки «Запланировано» нужно время запуска — его задают агенты, перетащить карточку сюда нельзя.'
  },
  arcRunning: 'Сейчас над задачей работает агент.',
  arcStale: 'Задача взята, но от воркера нет сигнала уже 2+ минуты — диспетчер вернёт её в очередь.',
  title: 'Канбан',
  orchestrationSettings: 'Настройки оркестрации',
  newTask: 'Новая задача',
  filterCards: 'Фильтр карточек…',
  noMatch: 'Нет задач по этим фильтрам',
  noTasks: 'На этой доске нет задач',
  open: 'Открыть',
  select: modifier => `Выделить (${modifier}-клик)`,
  deselect: 'Снять выделение',
  moveTo: label => `Переместить в «${label}»`,
  delete: 'Удалить',
  reviewChecking: 'Агент-рецензент проверяет выполненную работу.',
  attachedTip: name => `Профиль ${name} назначен — диспетчер передаст ему задачу на следующем цикле (≤1 мин).`,
  orchestratorTip: name => `${name} (оркестратор) возьмёт задачу на следующем цикле и напишет спецификацию.`,
  autoAssignTip: name =>
    `На следующем цикле диспетчера задача будет автоматически назначена профилю «${name}» (kanban.default_assignee).`,
  wontRun: 'не запустится',
  wontRunTip:
    'Карточки из «Готово к запуску» запускаются только после назначения профиля. Откройте карточку и выберите исполнителя или задайте исполнителя по умолчанию в настройках оркестрации.',
  noHeartbeat: 'нет сигнала',
  expand: label => `Развернуть колонку «${label}»`,
  collapse: label => `Свернуть колонку «${label}»`,
  newTaskIn: label => `Новая задача в колонке «${label}»`,
  empty: 'Пусто',
  unassigned: 'не назначен',
  filters: 'Фильтры',
  allProfiles: 'Все профили',
  allTenants: 'Все тенанты',
  showArchived: 'Показывать архив',
  groupRunning: 'Группировать «Выполняется» по профилям',
  nSelected: n => `Выделено: ${n}`,
  moveToShort: 'Переместить в',
  assign: 'Назначить',
  unassignAction: 'Снять исполнителя',
  archive: 'Архивировать',
  clearSelection: 'Снять выделение (Esc)',
  refused: 'отклонено',
  bulkFailed: (failed, total, err) =>
    `Не удалось: ${failed} из ${total} — ${err}. Карточки с ошибкой остаются выделенными.`,
  titlePlaceholderTriage: 'Сырая идея — агент её уточнит',
  titlePlaceholder: 'Название',
  descPlaceholder: 'Описание (необязательно)',
  priority: 'Приоритет',
  workspace: 'Рабочая папка',
  boardDefaultSuffix: ' · по умолчанию для доски',
  workspaceOverride: 'Свой путь к рабочей папке (необязательно)',
  model: 'Модель',
  modelInherit: 'Как в профиле',
  modelClear: 'Сбросить переопределение модели',
  modelHint:
    'Задача выполняется на выбранной модели с заданной глубиной рассуждений. Если не задано, используются настройки назначенного профиля.',
  workspaceInherit: 'Наследует папку проекта доски',
  workspaceInheritDir: dir => `Оставьте пустым, чтобы использовать ${dir}`,
  workspaceInheritGeneric: 'Оставьте пустым, чтобы использовать папку проекта доски.',
  assignee: 'Исполнитель',
  defaultOption: name => `${name} (по умолчанию)`,
  parkedOption: 'без исполнителя (отложена — не запустится)',
  skills: 'Навыки (через запятую)',
  skillsPlaceholder: 'translation, github',
  parent: 'Родительская задача (блокирует, пока не выполнена)',
  noParent: '— без родительской задачи —',
  goalMode: 'Режим цели (воркер повторяет попытки, пока агент-судья не признает задачу выполненной)',
  creating: 'Создание…',
  createTask: 'Создать задачу',
  cancel: 'Отмена',
  save: 'Сохранить',
  estimate: 'Оценка',
  estimateEffort: 'Оценить трудоёмкость',
  estimating: 'Оценка…',
  reEstimate: 'Оценить заново',
  makesModelCall: 'вызывает модель',
  estimateTip: 'Примерная оценка токенов и сложности от вспомогательной модели — вызывает модель.',
  estimateTipLong: 'Быстро вызывает вспомогательную модель, чтобы оценить токены и сложность. Это ориентир, а не счёт.',
  roughEstimate: 'Примерная оценка',
  tokUnit: 'ток',
  couldNotEstimate: 'Не удалось оценить',
  complexity: { S: 'Простая', M: 'Средняя', L: 'Сложная' },
  introBody:
    'Карточки запускаете не вы, а агенты. Переместите карточку в «Готово к запуску» и назначьте исполнителя — агент возьмёт её в течение минуты. Нет исполнителя — нет запуска. «Уточнение»: агент сначала превращает идею в полноценную задачу. «К выполнению»: ждёт других карточек. «Запланировано»: ждёт таймера. «Выполняется» и «Проверка»: здесь работают агенты, не вмешивайтесь. «Заблокировано»: ждёт вас. Результаты возвращаются на карточку.',
  introGotIt: 'Понятно',
  evtCreated: (where, assignee) =>
    `создана${where ? ` в колонке «${where}»` : ''}${assignee ? ` · исполнитель: ${assignee}` : ''}`,
  evtMovedTo: col => `перемещена в колонку «${col}»`,
  evtParentReopened: parent => `родительская задача ${parent} снова открыта`,
  evtAssignedTo: assignee => `назначен исполнитель ${assignee}`,
  evtUnassigned: 'исполнитель снят',
  evtCommentBy: author => `комментарий от ${author}`,
  evtClaimedReview: 'взята агентом-рецензентом',
  evtClaimedWorker: 'взята воркером',
  evtWorkerStarted: 'воркер запущен',
  evtCompleted: 'завершена',
  evtBlocked: 'заблокирована — нужен ответ человека',
  evtUnblocked: col => `разблокирована${col ? ` → ${col}` : ' → Готово к запуску'}`,
  evtReclaimed: 'отозвана у воркера — возвращена в очередь',
  evtSpecified: 'агент уточнения написал спецификацию',
  evtPromoted: 'зависимости выполнены — готова к запуску',
  evtScheduled: 'запланирована на потом',
  evtArchived: 'архивирована',
  evtReprioritized: priority => `приоритет изменён на ${priority}`,
  // Only ever the author slot of evtCommentBy («комментарий от …»), hence the genitive.
  someone: 'неизвестного автора',
  metaPriority: 'Приоритет',
  metaTenant: 'Тенант',
  metaCreatedBy: 'Автор',
  metaCreated: 'Создана',
  metaWorkerPid: 'PID воркера',
  blockKindTip: kind =>
    kind === 'needs_input'
      ? 'Воркер ждёт ответа от человека.'
      : kind === 'capability'
        ? 'Воркер упёрся в ограничение своих возможностей и не может его обойти.'
        : kind === 'transient'
          ? 'Воркер остановился из-за временной ошибки.'
          : 'Воркер сообщил о зависимости, которую нужно дождаться.',
  blockReason: 'Тип блокировки',
  blockRecurrences: 'Повторные блокировки',
  blockRecurrencesTip: 'Сколько раз задача снова блокировалась по той же причине после ручной разблокировки.',
  consecutiveFailures: 'Сбоев подряд',
  lastFailureError: 'Последний сбой',
  unblockedMessage: id => `Задача ${id} разблокирована и будет взята на следующем цикле.`,
  readyUnassignedTitle: 'Готова к запуску, но без исполнителя — эта карточка никогда не запустится.',
  readyUnassignedBody:
    'Диспетчер берёт из «Готово к запуску» только карточки с исполнителем. Выберите профиль в поле «Исполнитель» выше (или задайте исполнителя по умолчанию в настройках оркестрации) — и задача запустится в течение минуты.',
  diagnosticsN: n => `Диагностика · ${n}`,
  commandCopied: 'Команда скопирована',
  description: 'Описание',
  editDescription: 'Изменить описание',
  cancelEdit: 'Отменить изменения',
  noDescription: 'Описания пока нет.',
  result: 'Результат',
  latestSummary: 'Последняя сводка',
  dependencies: 'Зависимости',
  blockedBy: 'Зависит от',
  blocks: 'Блокирует',
  comments: n => `Комментарии · ${n}`,
  commentsHelpRunning:
    'Задача выполняется. Ваша заметка попадёт в текущий ход воркера за несколько секунд — без блокировки и разблокировки. «Вернуть в очередь с заметкой» вместо этого перезапустит задачу с нуля, и заметка будет в её контексте.',
  commentsHelp: 'Комментарии добавляются в ветку задачи. Воркер, взявший задачу, читает их как часть контекста.',
  send: 'Отправить',
  comment: 'Комментировать',
  messageWorker: 'Напишите работающему воркеру…',
  addComment: 'Добавьте комментарий…',
  deliveredLive: 'Дойдёт до работающего воркера за несколько секунд.',
  requeueWithNote: 'Вернуть в очередь с заметкой',
  notePosted: 'Заметка добавлена — задача снова в очереди',
  activity: n => `Активность · ${n}`,
  runs: n => `Запуски · ${n}`,
  workerLog: 'Журнал воркера',
  workerLogTail: 'Журнал воркера · последние строки',
  attachments: n => `Вложения · ${n}`,
  noAttachments: 'Вложений пока нет.',
  uploadAttachment: 'Загрузить вложение',
  taskActions: 'Действия с задачей',
  copyTaskId: 'Копировать ID задачи',
  copyTitle: 'Копировать название',
  copiedId: id => `Скопировано: ${id}`,
  copiedTitle: 'Название скопировано',
  close: 'Закрыть',
  working: 'в работе',
  board: 'Доска',
  newBoard: 'Новая доска',
  switchBoard: 'Сменить доску',
  pinBoard: 'Закрепить доску',
  unpinBoard: 'Открепить доску',
  pinnedBoards: 'Закреплённые доски',
  pinnedBoardsHint: 'Перетащите, чтобы изменить порядок, или нажмите Alt + ←/→',
  newBoardDots: 'Новая доска…',
  exportDots: 'Экспорт…',
  importDots: 'Импорт…',
  renameDots: 'Переименовать…',
  settingsDots: 'Настройки…',
  exportBoardTitle: 'Экспорт доски…',
  importBoardTitle: 'Импорт доски…',
  boardExported: path => `Доска экспортирована в ${path}`,
  boardImported: name => `Доска «${name}» импортирована`,
  boardImportedAs: slug => `Это имя уже занято — доска импортирована как ${slug}`,
  renameBoardTitle: 'Переименовать доску',
  deleteBoardTitle: name => `Удалить «${name}»?`,
  deleteBoardConfirm:
    'Доска будет перемещена в архив, а не стёрта: её задачи и вложения останутся на диске, и их можно восстановить.',
  boardArchived: path => `Доска перемещена в архив: ${path}`,
  boardSettingsFor: name => `Настройки доски — ${name}`,
  name: 'Название',
  boardNamePlaceholder: 'Название доски',
  slug: slug => `идентификатор: ${slug}`,
  project: 'Проект',
  noProject: 'Без проекта (временные песочницы)',
  projectHintPre:
    'Новые задачи выполняются в репозитории проекта (отдельный worktree на каждую задачу); при создании задачи рабочую папку всё равно можно переопределить. Управлять проектами можно командой ',
  projectHintCmd: 'hermes project',
  createBoard: 'Создать доску',
  orchestratorProfile: 'Профиль оркестратора',
  defaultAssignee: 'Исполнитель по умолчанию',
  defaultParen: '(по умолчанию)',
  autoDecompose: 'Автоматически разбивать задачи из «Уточнения»',
  profileDescriptions: 'Описания профилей',
  profileDescriptionsHint:
    'По описаниям агент разбиения решает, какому профилю отдать задачу. Сгенерируйте их вспомогательной моделью или напишите сами.',
  profileGoodAt: 'В чём силён этот профиль?',
  auto: 'Авто',
  notify: {
    completedTitle: 'Задача завершена',
    blockedTitle: 'Задача заблокирована — нужен ваш ответ',
    blockLoopTitle: 'Задача отправлена на уточнение — нужно решение',
    gaveUpTitle: 'Задача остановлена',
    gaveUpBody: 'Hermes не смог завершить эту задачу. Откройте канбан, чтобы узнать причину и переназначить её.',
    crashedTitle: 'С задачей возникла проблема — Hermes автоматически повторит попытку',
    timedOutTitle: 'Задача выполнялась слишком долго — Hermes автоматически повторит попытку',
    openKanban: 'Открыть канбан',
    artifacts: (n: number) => `${n} ${plural(n, 'артефакт', 'артефакта', 'артефактов')}`
  }
}
