import type { QueueFairCopy } from './types_queue_fair'

export const esQueueFair: QueueFairCopy = {
  holdUnknown: reason => `En espera: ${reason}`,
  hold: {
    session_busy: 'Esperando al turno actual',
    slot: 'Esperando un hueco de sesión libre',
    editing: 'En edición, no empezará todavía',
    restarting: 'Esperando a que se reinicie el gateway',
    paused: 'En pausa hasta resolver el turno perdido',
    provider_limit: 'Esperando a que se restablezca el límite del proveedor',
    memory: 'Esperando memoria libre'
  },
  moveUp: 'Subir',
  moveDown: 'Bajar',
  holdFailed: 'No se pudo pausar este mensaje en cola mientras lo editas; puede empezar antes de guardar'
}
