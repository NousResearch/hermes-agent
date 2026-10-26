import type { ClarifyCopy } from './types_clarify'

export const esClarify: Partial<ClarifyCopy> = {
  notReady: 'La solicitud de aclaración aún no está lista',
  gatewayDisconnected: 'Hermes está sin conexión ahora mismo. Reconéctate y vuelve a enviarlo.',
  sendFailed: 'No se pudo enviar la respuesta de aclaración',
  loadingQuestion: 'Cargando pregunta…',
  other: 'Otro (escribe tu respuesta)',
  placeholder: 'Escribe tu respuesta…',
  skip: 'Omitir',
  skipped: 'Omitido',
  noAnswer: 'Sin respuesta',
  confirmAndContinueLabel: 'Confirmar y continuar',
  singleSelectHint: 'Elige una',
  multiSelectHint: 'Elige todas las que correspondan',
  oneQuestion: '1 pregunta',
  questionProgress: (answered, total) => `${answered} de ${total} respondidas`,
  notDelivered:
    'Esta pregunta no llegó a la app, así que no se puede responder aquí. Pulsa Detener para terminar el turno y luego responde en el chat.'
}
