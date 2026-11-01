import type { QueueFairCopy } from './types_queue_fair'

export const ruQueueFair: QueueFairCopy = {
  holdUnknown: reason => `Ожидание: ${reason}`,
  hold: {
    session_busy: 'Ожидает завершения текущего хода',
    slot: 'Ожидает свободного слота сессии',
    editing: 'Редактируется, пока не начнётся',
    restarting: 'Ожидает перезапуска шлюза',
    paused: 'Приостановлено до разрешения потерянного хода',
    provider_limit: 'Ожидает сброса лимита провайдера',
    memory: 'Ожидает свободной памяти'
  },
  moveUp: 'Переместить вверх',
  moveDown: 'Переместить вниз',
  holdFailed: 'Не удалось приостановить это сообщение в очереди на время редактирования; оно может начаться до сохранения'
}
