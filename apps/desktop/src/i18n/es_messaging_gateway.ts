import type { MessagingGatewayTranslations } from './types_messaging_gateway'

export const esMessagingGateway = {
  hintGatewayStopped: 'Inicia aquí el gateway de mensajería para conectar.',
  restartNeeded: 'Guardado. Reinicia el gateway de mensajería para que la nueva configuración surta efecto.',
  restartNow: 'Reiniciar ahora',
  restarting: 'Reiniciando…',
  restartFailedManual: 'Hermes no pudo reiniciarse para aplicar tu configuración de mensajería',
  restartFailedManualDetail: 'Vuelve a pulsar Reiniciar; si sigue fallando, abre los registros y envía un diagnóstico.',
  startMessagingGateway: 'Iniciar gateway de mensajería',
  startingMessagingGateway: 'Iniciando gateway de mensajería…',
  gatewayStartFailed: 'No se pudo iniciar el gateway de mensajería.'
} satisfies MessagingGatewayTranslations
