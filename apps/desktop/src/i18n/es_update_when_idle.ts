import type { UpdateWhenIdleCopy } from './types_update_when_idle'

export const esUpdateWhenIdle: UpdateWhenIdleCopy = {
  busyTitle: 'Hermes sigue trabajando',
  busyBody: 'Actualizar reinicia Hermes, lo que detiene:',
  chats: (n: number) => (n === 1 ? '1 chat en curso' : `${n} chats en curso`),
  cron: (jobs: string) => `Tarea programada en curso (${jobs})`,
  delegation: 'Hay subagentes trabajando en segundo plano',
  input: 'Una pregunta espera tu respuesta',
  unknown: 'Hermes no pudo confirmar que está inactivo',
  updateNow: (n: number) => `Actualizar ahora (interrumpe ${n})`,
  updateWhenIdle: 'Actualizar cuando esté libre',
  waitingTitle: 'La actualización espera a que Hermes termine',
  waitingBody:
    'La actualización empieza sola cuando no haya nada en curso. Sigue trabajando; Hermes sigue disponible hasta entonces.',
  waitingFor: 'Esperando:',
  waitingIdle: 'No hay nada en curso. Iniciando la actualización…',
  waitingUntil: (time: string) => `Si Hermes sigue ocupado a las ${time}, se omite la actualización.`,
  cancel: 'Cancelar actualización',
  timedOut: 'Hermes siguió ocupado, así que la actualización no se ejecutó. No se cambió nada; inténtalo más tarde.'
}
