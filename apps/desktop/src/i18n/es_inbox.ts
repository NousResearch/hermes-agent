import type { InboxCopy } from './types_inbox'

export const esInbox: InboxCopy = {
  title: 'Bandeja de entrada',
  unreadCount: (count: number) => `${count} sin leer`,
  loading: 'Cargando la bandeja de entrada…',
  emptyTitle: 'Aún no hay resultados',
  emptyDesc: 'Cuando termina una tarea programada, su resultado llega aquí.',
  running: 'En curso',
  unread: 'Sin leer',
  noPreview: 'Terminó sin respuesta.',
  refresh: 'Actualizar',
  refreshing: 'Actualizando…',
  markReadFailed: 'No se pudo marcar el resultado como leído',
  newResultsTitle: (count: number) => `${count} resultados nuevos de tareas programadas`,
  newResultsBody: 'Abre la bandeja de entrada para leerlos.'
}
