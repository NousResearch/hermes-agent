import { mergeTranslations, type TranslationOverride } from '@hermes/shared/i18n'

import type { ManagedRolloutMessages } from '../managed-rollouts-types'

import { managedRolloutsEn } from './en'

const overrides: TranslationOverride<ManagedRolloutMessages> = {
  title: 'Управляемые развёртывания',
  noActive: 'Нет снимка активного развёртывания.',
  history: 'История развёртываний',
  unresolved: count => `Неразрешённые блокировки восстановления: ${count}`,
  archived: 'архивировано',
  active: 'активно',
  sections: {
    fleet: 'Установки управляемого развёртывания',
    preparation: 'Подготовка управляемого развёртывания',
    configuration: 'Настройка управляемого развёртывания',
    preflight: 'Предварительная проверка развёртывания',
    wavePreview: 'Предпросмотр волн развёртывания',
    active: 'Активное управляемое развёртывание',
    controls: 'Управление развёртыванием',
    recovery: 'Восстановление после развёртывания',
    history: 'История управляемых развёртываний',
    summary: 'Итоги развёртывания'
  },
  actions: {
    select: 'Выбрать',
    selected: 'Выбрано',
    prepare: 'Подготовить выбранные установки',
    preparing: 'Подготовка…',
    recheckEligibility: 'Повторно проверить пригодность',
    continueToPreflight: 'Перейти к предварительной проверке',
    start: 'Начать развёртывание',
    starting: 'Запуск…',
    pause: 'Приостановить',
    pausing: 'Приостановка…',
    stop: 'Остановить',
    stopping: 'Остановка…',
    resume: 'Возобновить',
    verifyBeforePromotion: 'Проверить перед переходом к следующей волне',
    verifyingBeforePromotion: 'Проверка перед переходом к следующей волне',
    recheckOutcome: 'Повторно проверить исход',
    recoverConnections: 'Восстановить подключения',
    retry: 'Повторить попытку',
    exclude: 'Исключить',
    stopAndPlanRetry: 'Остановить и запланировать повторную попытку',
    archiveStoppedRollout: 'Архивировать остановленное развёртывание'
  },
  status: {
    activePhase: phase => `Текущий этап: ${phase}`,
    queued: 'В очереди',
    preparing: 'Подготовка',
    ready: 'Готово',
    running: 'Выполняется',
    awaitingPromotion: 'Ожидает перехода к следующей волне',
    attentionRequired: 'Требуется внимание',
    paused: 'Приостановлено',
    stopped: 'Остановлено',
    completed: 'Завершено',
    completedWithExclusions: 'Завершено с исключениями',
    failed: 'Ошибка',
    refused: 'Отказано',
    unknown: 'Исход неизвестен',
    unverified: 'Не подтверждено',
    recoveryRequired: 'Требуется восстановление',
    fenced: 'Защитная блокировка восстановления сохранена',
    alreadyCurrent: 'Уже установлена нужная версия',
    pending: 'Ожидаются подтверждения'
  },
  policy: {
    manual: 'Ручное подтверждение',
    automatic: 'Автоматическое продолжение после подтверждения канарейки при исправном состоянии',
    mode: mode => `Режим: ${mode}`,
    canaryGate: gate => `Проверка канарейки: ${gate}`
  },
  labels: {
    progressionMode: 'Режим перехода между волнами',
    concurrency: 'Параллельность',
    inventoryRevision: (revision, capturedMono) => `Ревизия инвентаризации: ${revision}; монотонное время снимка: ${capturedMono}.`,
    targetDetails: (label, alias, machineId, installId) =>
      `${label} · ${alias ? `${alias} · ` : ''}${machineId} · ${installId}`,
    wave: (number, targets) => `Волна ${number}: ${targets || 'нет установок'}`,
    emptyWave: number => `Волна ${number}: нет установок`,
    progress: (completed, total) => `Ход выполнения: завершено ${completed} из ${total} установок.`,
    receipt: (outcome, correlationId) => `Квитанция: ${outcome} (${correlationId})`,
    readiness: value => `Готовность: ${value}`,
    reason: value => `Причина: ${value}`,
    attempt: (installId, phase) => `${installId} · ${phase}`,
    historyEntry: (id, phase, updatedAt, unresolved, archived) =>
      `${id} · ${phase} · ${updatedAt} · неразрешённых блокировок: ${unresolved} · ${archived ? 'архивировано' : 'активно'}`,
    archive: archived => `Архив: ${archived ? 'архивировано' : 'активно'}`
  },
  warnings: {
    sharedMachine: 'Несколько установок находятся на одной машине; проверьте владельца каждой перед подготовкой.',
    unsupportedTarget: 'Эта установка не поддерживает управляемое развёртывание.',
    preparation: 'Подготовка следует за текущей вершиной настроенной ветки; она не входит в развёртывание закреплённой версии.',
    preparationMayDisconnect: 'Во время подготовки сеансы могут временно потерять соединение.',
    preparationInvalidatesReview:
      'Подготовка делает прежнюю проверку недействительной. После изменения цели, псевдонима, источника или области действия требуется повторная проверка.',
    changedPlan: 'Конфигурация изменилась; перед продолжением обновите подтверждение плана.',
    projectionUnavailable: 'Предпросмотр волн недоступен; перейти к предварительной проверке нельзя.',
    incompatible: 'Текущие возможности несовместимы с планом; начать развёртывание нельзя.',
    commitmentBoundary:
      'Остановка запрещает новые разрешения на запуск, но уже разрешённые обновления могут продолжить отключение сеансов, применение версии и восстановление областей конфигурации.',
    unknownOutcome: 'Исход на удалённой установке неизвестен; отсутствие квитанции не доказывает успешное обновление.',
    recoveryFence: 'Защитная блокировка восстановления сохранена; новая небезопасная операция остаётся запрещённой.',
    unknownAndFenced:
      'Исход неизвестен, и защитная блокировка восстановления сохранена; новая небезопасная операция остаётся запрещённой.',
    estimateUnavailable: 'Оценка времени недоступна.',
    unavailable: 'Данные управляемого развёртывания недоступны. Новая операция не запускалась.',
    stale: 'Снимок развёртывания может быть устаревшим. Перед действием выполните повторную проверку.',
    reconnecting: 'Восстанавливается соединение с состоянием управляемого развёртывания…'
  },
  descriptions: {
    inventoryObserved: 'Инвентаризация отражает только наблюдаемое состояние. Выбор цели сам по себе не подтверждает её пригодность и не разрешает обновление.',
    serialCapability:
      'Контракт выбранной версии требует последовательного обновления. Выбранная канарейка остаётся прежней, пока черновик плана не изменится.',
    canonicalPlan:
      'Отправленный черновик точно соответствует показанному плану волн. Обновлённые или изменённые строки нужно подтвердить снова.',
    manualPolicy: 'После завершения каждой волны требуется ручное подтверждение.',
    automaticPolicy:
      'Автоматическое продолжение допускается только после явного подтверждения канарейки и получения свежих подтверждений при исправном состоянии развёртывания.',
    commitmentBoundary:
      'Уже разрешённое обновление может завершиться после остановки развёртывания. Остановка запрещает новые разрешения, но не отменяет работу, уже переданную удалённому обновляющему процессу.',
    lastSettledLocalFenceNotLive:
      'Показано последнее подтверждённое состояние вместе с локальной защитной блокировкой, а не текущее состояние удалённой установки. Изменения вне наблюдения Desktop не доказаны.',
    archiveRetainsFence:
      'Архивирование меняет только отображение и историю; оно не удаляет подтверждения и не снимает неразрешённую защитную блокировку.',
    noAutomaticResume:
      'После перезапуска сначала восстановите картину событий, затем явно возобновите развёртывание или разрешите переход к следующей волне. Сохранённая политика не означает нового разрешения.',
    historicalEstimate:
      'Оценки по прошлым запускам носят справочный характер и не разрешают запуск новых операций.'
  },
  summary: {
    outcome: phase => `Исход: ${phase}`,
    excludedTargets: count => `Исключено установок: ${count}`,
    unresolvedFences: count => `Неразрешённых блокировок восстановления: ${count}`,
    archive: archived => `Архив: ${archived ? 'архивировано' : 'активно'}`,
    reason: reason => `Причина: ${reason}`
  },
  a11y: {
    confirmPreparation: 'Подтверждаю, что эти установки составляют отдельный набор для подготовки.',
    confirmPreflight: 'Подтверждаю результаты предварительной проверки.',
    selectTarget: label => `Выбрать установку ${label}`,
    selectedTarget: label => `Снять выбор с установки ${label}`,
    warning: message => `Предупреждение: ${message}`,
    status: message => `Состояние: ${message}`,
    action: message => `Действие: ${message}`,
    historyEntry: id => `Открыть сведения о развёртывании ${id}`,
    attemptToggle: (installId, expanded) =>
      `${expanded ? 'Свернуть' : 'Развернуть'} сведения о попытке для установки ${installId}`
  }
}

export const managedRolloutsRu = mergeTranslations<ManagedRolloutMessages>(managedRolloutsEn, overrides)
export const managedRollouts = managedRolloutsRu
export default managedRolloutsRu
