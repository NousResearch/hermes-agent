// Desktop-side /rollback /agents /config /history /save /export /import /timestamps lines
// (en_desktop_commands.ts is the source).
export const ruDesktopCommands = {
  error: (message: string) => `ошибка: ${message}`,
  rollbackUsage: 'использование: /rollback [list] | diff <n|hash> | <n|hash> [file] [--all]',
  checkpointsDisabled: 'Контрольные точки для этого сеанса не включены (задайте checkpoints.enabled в config.yaml).',
  checkpointsEmpty: 'Для этого рабочего каталога ещё нет контрольных точек.',
  checkpointsHeader: 'Контрольные точки (сначала новые):',
  checkpointsHint:
    'Восстановление: /rollback <n> [file] (с --all перезаписываются и ваши правки); предпросмотр: /rollback diff <n>.',
  noDiff: 'С этой контрольной точки изменений нет.',
  rollbackFailed: (error: string) => `Откат не удался: ${error}`,
  unknownError: 'неизвестная ошибка',
  restored: (hash: string) => (hash ? `Контрольная точка ${hash} восстановлена.` : 'Контрольная точка восстановлена.'),
  rewound: (count: string) => `Из этого разговора откачено сообщений: ${count}.`,
  reopenForRewind: 'Откройте этот чат заново, чтобы увидеть откаченный разговор.',
  agentsEmpty: 'В этом чате нет активных субагентов и фоновых процессов.',
  subagentsHeader: (count: string) => `Субагенты (${count}):`,
  subagentFallback: 'субагент',
  statusRunning: 'выполняется',
  toolCalls: (count: string) => `вызовов инструментов: ${count}`,
  processesHeader: (count: string) => `Фоновые процессы (${count}):`,
  configHeader: 'Конфигурация (настройки отображения, подтверждений, вставки и голоса; остальное не показывается):',
  configDefaults: '(по умолчанию)',
  configFooter:
    'Меняйте настройки в «Настройках» или редактируйте config.yaml командой `hermes config edit` на хосте шлюза.',
  historyHeader: (count: string) => `История разговора (сообщений: ${count}):`,
  historyEmpty: 'Сообщений пока нет.',
  historyYou: 'Вы',
  historyHermes: 'Hermes',
  saveUsage: 'использование: /save [json|md|html] [filename] [redact]',
  saveLegacyRefused:
    'Не сохранено: этот шлюз не умеет экспортировать скрытую, именованную или не-JSON копию. Обновите шлюз или выполните /save без параметров, чтобы записать исходный JSON на хосте шлюза.',
  savedOnHost: (path: string) => `Сохранено на хосте шлюза: ${path}`,
  unknownPath: '(путь неизвестен)',
  saveUnsupported: 'ошибка: эта сборка приложения не умеет сохранять файлы',
  saveDialogTitle: 'Сохранить разговор',
  savedTo: (path: string) => `Разговор сохранён в ${path}`,
  saveCancelled: 'Не сохранено (отменено).',
  exportRemote:
    '/export упаковывает профиль с хоста шлюза, а это не этот компьютер. Выполните `hermes profile export` на том хосте.',
  importRemote:
    '/import устанавливает профиль на хост шлюза, а это не этот компьютер. Выполните `hermes profile import <archive>` на том хосте.',
  timestampsUsage: 'использование: /timestamps [on|off|status]',
  timestampsOn: 'Метки времени: вкл. (только это окно; чтобы сохранить, задайте display.timestamps в config.yaml)',
  timestampsOff: 'Метки времени: выкл. (только это окно; чтобы сохранить, задайте display.timestamps в config.yaml)'
}
