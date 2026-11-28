/**
 * Russian bundle for the Kanban plugin. Same message SHAPE as `en`
 * (KanbanMessages); registered in KANBAN_LOCALES. Technical identifiers
 * (profile names, task ids, paths, raw statuses) are never translated here —
 * they come from the backend and are rendered as-is by the components.
 *
 * Deterministic only: diagnostics/errors are mapped by kind / regex to a fixed
 * Russian explanation, and the untouched original is kept by the UI in a
 * collapsible «Технические подробности» block. No LLM translation of logs.
 */

import type { KanbanMessages } from './i18n'

// Plugins may not import core modules, so the Russian plural rule (1 раз / 2–4 раза / 5+ раз) lives here.
const raz = (n: number) => {
  const d = Math.abs(n) % 10
  const dd = Math.abs(n) % 100

  return `${n} ${d >= 2 && d <= 4 && (dd < 12 || dd > 14) ? 'раза' : 'раз'}`
}

const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined)

const humanAge = (seconds: number) => {
  const m = Math.round(seconds / 60)

  if (m < 60) {
    return `${m} мин`
  }

  const h = Math.round(m / 60)

  return h < 48 ? `${h} ч` : `${Math.round(h / 24)} дн`
}

// Fixed explanations for the diagnostics the dispatcher raises (kanban_diagnostics.py).
const DIAG_TITLE: Record<string, (title: string, count: number, data: Record<string, unknown>) => string> = {
  repeated_crashes: (title, count, data) => {
    const n = num(data.consecutive_crashes) ?? Number(/crashed (\d+)x/.exec(title)?.[1] ?? count)

    return `Агент аварийно завершился ${raz(n)}`
  },
  repeated_failures: (title, count) => {
    if (/Provider rejected/i.test(title)) {
      return 'Провайдер отклонил ключ или модель профиля — блокировка после одной попытки'
    }

    const n = Number(/x(\d+)/.exec(title)?.[1] ?? count)

    return `Агент завершился с ошибкой ${raz(n)} подряд`
  },
  hallucinated_cards: () => 'Воркер сослался на несуществующие карточки',
  triage_aux_unavailable: () => 'Для разбора нет доступной модели',
  prose_phantom_refs: () => 'В итоговом описании указаны неизвестные ID задач',
  review_dependency_deadlock: title => {
    const n = Number(/(\d+)/.exec(title)?.[1] ?? 0)

    return `Передача на проверку блокирует зависимые задачи: ${n}`
  },
  running_with_open_parents: title => {
    const n = Number(/(\d+)/.exec(title)?.[1] ?? 0)

    return `Задача выполняется, хотя родительских задач без завершения: ${n}`
  },
  stuck_in_blocked: (title, _c, data) => {
    const h = num(data.age_hours) ?? Number(/(\d+)h/.exec(title)?.[1] ?? 0)

    return `Задача заблокирована уже ${Math.round(h)} ч`
  },
  block_unblock_cycling: (title, _c, data) => {
    const cycles = num(data.cycles) ?? Number(/cycled (\d+)x/.exec(title)?.[1] ?? 0)

    return `Блокировка и разблокировка повторились ${raz(cycles)}`
  },
  stranded_in_ready: (_t, _c, data) => {
    const s = num(data.age_seconds)

    return s === undefined ? 'Задача готова к запуску, но воркера нет' : `Готова к запуску ${humanAge(s)}, воркера нет`
  },
  stale_running: () => 'Задача «выполняется», но аренда истекла и живого воркера нет'
}

const DIAG_EXPLAIN: Record<string, string> = {
  repeated_crashes:
    'Процесс агента несколько раз подряд завершился с ошибкой, и диспетчер остановил автоматические повторы, чтобы не тратить токены. Посмотрите «Технические подробности» и журнал воркера; повторный запуск имеет смысл после устранения причины.',
  repeated_failures:
    'Агент несколько раз подряд не смог выполнить задачу. Автоматические повторы остановлены. Проверьте ключ, модель и описание задачи, затем разблокируйте или переназначьте её.',
  hallucinated_cards:
    'В ответе воркера названы карточки, которых нет на доске. Результату доверять нельзя: проверьте его вручную.',
  triage_aux_unavailable:
    'Для автоматического разбора задач не настроена рабочая вспомогательная модель. Назначьте исполнителя вручную или настройте модель.',
  prose_phantom_refs:
    'В итоговом описании упомянуты ID задач, которых нет на доске. Проверьте текст результата.',
  review_dependency_deadlock:
    'Задача на проверке не может завершиться, пока зависимые задачи ждут её результата. Нужно решение человека.',
  running_with_open_parents:
    'Задача запущена до завершения родительских задач — её результат может опираться на неполные данные.',
  stuck_in_blocked:
    'Задача долго ждёт вас. Откройте карточку, прочитайте причину блокировки и ответьте комментарием или разблокируйте.',
  block_unblock_cycling:
    'Задачу несколько раз разблокировали, но она снова блокировалась по той же причине. Устраните причину, а не просто разблокируйте.',
  stranded_in_ready:
    'Задача готова, но её никто не берёт: нет исполнителя или он недоступен. Назначьте исполнителя.',
  stale_running:
    'Воркер перестал отвечать, а запись о запуске осталась. Верните задачу в очередь кнопкой «Освободить».'
}

// Known raw technical fragments → short Russian meaning. Order matters; all matches are joined.
const ERROR_PATTERNS: Array<[RegExp, (m: RegExpExecArray) => string]> = [
  [/Agent crash(?:ed)?\s*x?(\d+)/i, m => `Агент аварийно завершился ${raz(Number(m[1]))}`],
  [/managed_worker_failed/, () => 'Ошибка управляемого воркера (managed_worker_failed)'],
  [/Kanban owner refused/i, () => 'Владелец сессии отклонил запрос'],
  [/Kanban execution unknown/i, () => 'Итог запуска неизвестен: выполнение осталось за владельцем (Gateway)'],
  [
    /couldn.t save this conversation/i,
    () => 'Не удалось сохранить диалог воркера, поэтому он остановился без потери сообщений'
  ],
  [/Circuit breaker/i, () => 'Автоматические повторы остановлены после нескольких ошибок'],
  [/exited with code (\d+)/i, m => `Процесс завершился с кодом ${m[1]}`],
  [/timed out|timeout/i, () => 'Превышено время выполнения'],
  [/rate.?limit|429/i, () => 'Достигнут лимит запросов провайдера'],
  [/credential|api key|unauthorized|401|403/i, () => 'Провайдер отклонил ключ или доступ']
]

const OUTCOME: Record<string, string> = {
  completed: 'Завершён',
  crashed: 'Аварийно завершён',
  failed: 'Ошибка',
  timed_out: 'Превышено время',
  gave_up: 'Повторы остановлены',
  blocked: 'Заблокирован',
  running: 'Выполняется',
  reclaimed: 'Освобождён',
  done: 'Готово'
}

export const ru: KanbanMessages = {
  nav: 'Канбан',
  openBoard: 'Канбан: открыть доску',
  newTaskCommand: 'Канбан: новая задача',
  countTip: (running, ready) => `Канбан — выполняется: ${running}, готово к запуску: ${ready}`,
  col: {
    triage: { label: 'Разбор', help: 'Сырые идеи — агент-спецификатор превратит их в нормальную задачу.' },
    todo: { label: 'К выполнению', help: 'Ждёт другие задачи или исполнителя.' },
    scheduled: { label: 'Запланировано', help: 'Ждёт назначенного времени.' },
    ready: {
      label: 'Готово к запуску',
      help: 'Зависимости выполнены — назначьте исполнителя, и диспетчер запустит задачу.'
    },
    running: { label: 'Выполняется', help: 'Задачу взял воркер — над ней работает агент. Ставит диспетчер.' },
    blocked: {
      label: 'Заблокировано',
      help: 'Задача не может продолжить выполнение. Откройте диагностику, чтобы узнать причину.'
    },
    review: { label: 'На проверке', help: 'Агент-проверяющий смотрит результат. Ставит диспетчер.' },
    done: { label: 'Завершено', help: 'Выполнено; зависимые задачи становятся готовыми к запуску.' },
    archived: { label: 'Архив', help: 'Скрыто из обычного вида доски.' }
  },
  locked: {
    review: 'Статус «На проверке» ставит диспетчер, когда карточку берёт агент-проверяющий.',
    running: 'Статус «Выполняется» ставит диспетчер, когда воркер берёт карточку.',
    scheduled: 'Для «Запланировано» нужно время запуска — его задают агенты, перетащить карточку нельзя.'
  },
  arcRunning: 'Сейчас над задачей работает агент.',
  arcStale: 'Задача взята, но сигнала от воркера нет больше 2 минут — диспетчер освободит её.',
  title: 'Канбан',
  orchestrationSettings: 'Настройки оркестрации',
  newTask: 'Новая задача',
  filterCards: 'Фильтр карточек…',
  noMatch: 'Нет задач по фильтрам',
  noTasks: 'На этой доске нет задач',
  open: 'Открыть',
  select: modifier => `Выбрать (${modifier}+клик)`,
  deselect: 'Снять выбор',
  moveTo: label => `Переместить в «${label}»`,
  delete: 'Удалить',
  reviewChecking: 'Агент-проверяющий проверяет готовую работу.',
  attachedTip: name => `${name} назначен — диспетчер передаст задачу на следующем такте (до 1 мин).`,
  orchestratorTip: name => `${name} (оркестратор) возьмёт задачу на следующем такте и напишет спецификацию.`,
  autoAssignTip: name => `Автоматически назначится на «${name}» (kanban.default_assignee) на следующем такте.`,
  wontRun: 'не запустится',
  wontRunTip:
    'Готовые карточки запускаются только после назначения исполнителя. Откройте карточку и выберите исполнителя или задайте исполнителя по умолчанию в настройках оркестрации.',
  noHeartbeat: 'нет сигнала',
  expand: label => `Развернуть: ${label}`,
  collapse: label => `Свернуть: ${label}`,
  newTaskIn: label => `Новая задача в «${label}»`,
  empty: 'Пусто',
  unassigned: 'не назначен',
  filters: 'Фильтры',
  allProfiles: 'Все профили',
  allTenants: 'Все арендаторы',
  showArchived: 'Показать архив',
  groupRunning: 'Группировать «Выполняется» по профилю',
  nSelected: n => `Выбрано: ${n}`,
  moveToShort: 'Переместить',
  assign: 'Назначить',
  unassignAction: 'Снять исполнителя',
  archive: 'В архив',
  clearSelection: 'Снять выбор (Esc)',
  refused: 'отказ',
  bulkFailed: (failed, total, err) => `Не удалось: ${failed} из ${total} — ${err}. Неудачные карточки остаются выбранными.`,
  titlePlaceholderTriage: 'Грубая идея — спецификатор доработает',
  titlePlaceholder: 'Название',
  descPlaceholder: 'Описание (необязательно)',
  priority: 'Приоритет',
  workspace: 'Рабочая папка',
  boardDefaultSuffix: ' · по умолчанию для доски',
  workspaceOverride: 'Путь к рабочей папке (необязательно)',
  model: 'Модель',
  modelInherit: 'По умолчанию из профиля',
  modelClear: 'Сбросить модель',
  modelHint:
    'Запускает эту задачу на выбранной модели и глубине рассуждения. Если не задано — берётся из профиля исполнителя.',
  workspaceInherit: 'Наследует проектную папку доски',
  workspaceInheritDir: dir => `Оставьте пустым, чтобы использовать ${dir}`,
  workspaceInheritGeneric: 'Оставьте пустым, чтобы использовать проектную папку доски.',
  assignee: 'Исполнитель',
  defaultOption: name => `${name} (по умолчанию)`,
  parkedOption: 'не назначен (отложена — не запустится)',
  skills: 'Навыки (через запятую)',
  skillsPlaceholder: 'перевод, github',
  parent: 'Родитель (ждёт его завершения)',
  noParent: '— без родителя —',
  goalMode: 'Режим цели (воркер работает, пока судья не признает задачу выполненной)',
  creating: 'Создаю…',
  createTask: 'Создать задачу',
  cancel: 'Отмена',
  save: 'Сохранить',
  estimate: 'Оценка',
  estimateEffort: 'Оценить сложность',
  estimating: 'Оцениваю…',
  reEstimate: 'Оценить заново',
  makesModelCall: 'обращается к модели',
  estimateTip: 'Примерная оценка токенов и сложности от вспомогательной модели — обращается к модели.',
  estimateTipLong:
    'Быстрый запрос к вспомогательной модели: оценка токенов и сложности. Это ориентир, а не счёт.',
  roughEstimate: 'Примерная оценка',
  tokUnit: 'ток.',
  couldNotEstimate: 'Не удалось оценить',
  complexity: { S: 'Малая', M: 'Средняя', L: 'Большая' },
  introBody:
    'Карточки запускаете не вы — их выполняют агенты. Переведите карточку в «Готово к запуску» с исполнителем, и агент возьмёт её в течение минуты. Нет исполнителя — нет запуска. Разбор: агент сначала превращает идею в нормальную задачу. К выполнению: ждёт другие карточки. Запланировано: ждёт времени. Выполняется и На проверке — зона агентов, не трогайте. Заблокировано: задача ждёт вас. Результаты приходят в карточку.',
  introGotIt: 'Понятно',
  evtCreated: (where, assignee) =>
    `создана${where ? ` в «${where}»` : ''}${assignee ? ` · исполнитель: ${assignee}` : ''}`,
  evtMovedTo: col => `перемещена в «${col}»`,
  evtParentReopened: parent => `родительская задача ${parent} открыта заново`,
  evtAssignedTo: assignee => `назначен исполнитель: ${assignee}`,
  evtUnassigned: 'исполнитель снят',
  evtCommentBy: author => `комментарий от ${author}`,
  evtClaimedReview: 'взята агентом-проверяющим',
  evtClaimedWorker: 'взята воркером',
  evtWorkerStarted: 'воркер запущен',
  evtCompleted: 'завершена',
  evtBlocked: 'заблокирована — нужен ответ человека',
  evtUnblocked: col => `разблокирована${col ? ` → ${col}` : ' → Готово к запуску'}`,
  evtReclaimed: 'освобождена — возвращена в очередь',
  evtSpecified: 'спецификацию написал агент разбора',
  evtPromoted: 'зависимости выполнены — переведена в «Готово к запуску»',
  evtScheduled: 'запланирована на потом',
  evtArchived: 'отправлена в архив',
  evtReprioritized: priority => `приоритет: ${priority}`,
  someone: 'кто-то',
  metaPriority: 'Приоритет',
  metaTenant: 'Арендатор',
  metaCreatedBy: 'Создал',
  metaCreated: 'Создано',
  metaWorkerPid: 'PID воркера',
  blockKindTip: kind =>
    kind === 'needs_input'
      ? 'Воркер попросил ответа от человека.'
      : kind === 'capability'
        ? 'Воркер упёрся в ограничение возможностей и не может продолжить.'
        : kind === 'transient'
          ? 'Воркер остановился из-за временной ошибки.'
          : 'Воркер сообщил о зависимости, которую нужно дождаться.',
  blockReason: 'Тип блокировки',
  blockRecurrences: 'Повторов блокировки',
  blockRecurrencesTip: 'Сколько раз задача снова блокировалась по той же причине после разблокировки человеком.',
  consecutiveFailures: 'Ошибок подряд',
  lastFailureError: 'Последняя ошибка',
  unblockedMessage: id => `Задача ${id} разблокирована и готова к следующему такту.`,
  readyUnassignedTitle: 'Готова к запуску, но без исполнителя — карточка не запустится.',
  readyUnassignedBody:
    'Диспетчер берёт только готовые карточки с исполнителем. Выберите профиль в поле «Исполнитель» выше (или задайте исполнителя по умолчанию в настройках оркестрации) — запуск произойдёт в течение минуты.',
  diagnosticsN: n => `Диагностика · ${n}`,
  commandCopied: 'Команда скопирована',
  description: 'Описание',
  editDescription: 'Изменить описание',
  cancelEdit: 'Отменить правку',
  noDescription: 'Описания пока нет.',
  result: 'Результат',
  latestSummary: 'Последний итог',
  dependencies: 'Зависимости',
  blockedBy: 'Зависит от задачи',
  blocks: 'Блокирует',
  comments: n => `Комментарии · ${n}`,
  commentsHelpRunning:
    'Задача выполняется. Ваша заметка попадёт в текущий ход воркера в течение нескольких секунд — без блокировки и разблокировки. «Перезапустить с заметкой» запускает задачу с нуля, добавив вашу заметку в контекст.',
  commentsHelp:
    'Комментарии добавляются в ветку задачи. Когда воркер берёт задачу, он читает их как часть контекста.',
  send: 'Отправить',
  comment: 'Комментарий',
  messageWorker: 'Сообщение работающему воркеру…',
  addComment: 'Добавить комментарий…',
  deliveredLive: 'Доставлено работающему воркеру в течение нескольких секунд.',
  requeueWithNote: 'Перезапустить с заметкой',
  notePosted: 'Заметка отправлена — задача возвращена в очередь',
  activity: n => `История · ${n}`,
  runs: n => `Запуски · ${n}`,
  workerLog: 'Журнал воркера',
  workerLogTail: 'Журнал воркера · конец',
  attachments: n => `Вложения · ${n}`,
  noAttachments: 'Вложений пока нет.',
  uploadAttachment: 'Загрузить вложение',
  taskActions: 'Действия с задачей',
  copyTaskId: 'Копировать ID задачи',
  copyTitle: 'Копировать название',
  copiedId: id => `Скопировано: ${id}`,
  copiedTitle: 'Название скопировано',
  close: 'Закрыть',
  working: 'работает',
  board: 'Доска',
  newBoard: 'Новая доска',
  switchBoard: 'Переключить доску',
  pinBoard: 'Закрепить доску',
  unpinBoard: 'Открепить доску',
  pinnedBoards: 'Закреплённые доски',
  pinnedBoardsHint: 'Перетащите или нажмите Alt + Влево/Вправо, чтобы изменить порядок',
  newBoardDots: 'Новая доска…',
  exportDots: 'Экспорт…',
  importDots: 'Импорт…',
  renameDots: 'Переименовать…',
  settingsDots: 'Настройки…',
  exportBoardTitle: 'Экспорт доски…',
  importBoardTitle: 'Импорт доски…',
  boardExported: path => `Доска экспортирована: ${path}`,
  boardImported: name => `Импортировано: ${name}`,
  boardImportedAs: slug => `Такое имя занято — импортировано как ${slug}`,
  renameBoardTitle: 'Переименовать доску',
  deleteBoardTitle: name => `Удалить «${name}»?`,
  deleteBoardConfirm: 'Доска будет архивирована, а не стёрта: задачи и вложения останутся на диске и их можно восстановить.',
  boardArchived: path => `Доска архивирована: ${path}`,
  boardSettingsFor: name => `Настройки доски — ${name}`,
  name: 'Название',
  boardNamePlaceholder: 'Название доски',
  slug: slug => `slug: ${slug}`,
  project: 'Проект',
  noProject: 'Без проекта (временные песочницы)',
  projectHintPre:
    'Новые задачи запускаются в репозитории проекта (отдельное рабочее дерево на задачу); для каждой задачи папку можно переопределить при создании. Проектами управляет команда ',
  projectHintCmd: 'hermes project',
  createBoard: 'Создать доску',
  orchestratorProfile: 'Профиль-оркестратор',
  defaultAssignee: 'Исполнитель по умолчанию',
  defaultParen: '(по умолчанию)',
  autoDecompose: 'Автоматически разбивать задачи из «Разбор»',
  profileDescriptions: 'Описания профилей',
  profileDescriptionsHint:
    'Описания подсказывают автоматическому разбиению, кому что поручать. Сгенерируйте их вспомогательной моделью или напишите сами.',
  profileGoodAt: 'В чём силён этот профиль?',
  auto: 'Авто',
  notify: {
    completedTitle: 'Задача завершена',
    blockedTitle: 'Задача заблокирована — нужен ваш ответ',
    blockLoopTitle: 'Задача отправлена на разбор — нужно решение',
    gaveUpTitle: 'Задача остановлена',
    gaveUpBody: 'Hermes не смог завершить эту задачу. Откройте Канбан, чтобы узнать причину и переназначить её.',
    crashedTitle: 'С задачей возникла проблема — Hermes повторит её автоматически',
    timedOutTitle: 'Задача выполнялась слишком долго — Hermes повторит её автоматически',
    openKanban: 'Открыть Канбан',
    artifacts: (n: number) => `Артефактов: ${n}`
  },
  techDetails: 'Технические подробности',
  diagTitle: (kind, title, count, data) => DIAG_TITLE[kind]?.(title, count, data) ?? '',
  diagExplain: kind => DIAG_EXPLAIN[kind] ?? '',
  errorExplain: raw => {
    const found: string[] = []

    for (const [re, say] of ERROR_PATTERNS) {
      const m = re.exec(raw)

      if (m) {
        found.push(say(m))
      }
    }

    return [...new Set(found)].join('. ')
  },
  runOutcome: outcome => OUTCOME[outcome] ?? outcome,
  blockKindName: kind =>
    ({ needs_input: 'Нужен ответ', capability: 'Ограничение возможностей', transient: 'Временная ошибка', dependency: 'Ожидание зависимости' })[
      kind
    ] ?? kind
}
