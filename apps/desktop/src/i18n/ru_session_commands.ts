// Typed /undo /retry /steer /queue lines (en_session_commands.ts is the source).
export const ruSessionCommands = {
  undoUsage: 'использование: /undo [N]',
  undoNothing: 'нечего отменять',
  undidOne: (count: string) => `отменено сообщений: ${count} — удалённое сообщение в поле ввода`,
  undidOther: (count: string) => `отменено сообщений: ${count} — удалённое сообщение в поле ввода`,
  retryNothing: 'нечего повторять',
  retryUnsupported: '/retry: нельзя повторить медиа или неизвестные части содержимого',
  busy: (command: string) =>
    `сеанс занят — Hermes ещё отвечает. Сначала остановите текущий ответ (кнопка «Стоп» или Ctrl+C в терминале), затем выполните /${command}.`,
  steerUsage: 'использование: /steer <запрос>',
  steerQueued: (preview: string) => `уточнение в очереди — придёт после следующего вызова инструмента: «${preview}»`,
  steerNoActiveTurn: (preview: string) => `нет активного хода — поставлено в очередь на следующий: «${preview}»`,
  steerRejected: 'уточнение отклонено — нет активного хода, поставлено в очередь на следующий ход',
  queued: (preview: string) => `в очереди: «${preview}»`,
  queueFailed: 'Не удалось поставить сообщение в очередь',
  countOne: (count: string) => `сообщений в очереди: ${count}`,
  countOther: (count: string) => `сообщений в очереди: ${count}`,
  queueEditKeptTitle: 'Сообщение в очереди не изменено',
  queueEditKept: 'Оно началось или было отклонено до сохранения правки. Отредактированный текст остался в поле ввода.',
  queueEditUnconfirmedTitle: 'Правка не подтверждена',
  queueEditUnconfirmed:
    'Hermes не смог подтвердить правку. Сохраните ещё раз, чтобы повторить ту же правку; дважды в очередь она не попадёт.'
}
