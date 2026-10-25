import type { MessagingGatewayTranslations } from './types_messaging_gateway'

export const deMessagingGateway = {
  hintGatewayStopped: 'Starten Sie das Messaging-Gateway hier, um die Verbindung herzustellen.',
  restartNeeded: 'Gespeichert. Starten Sie das Messaging-Gateway neu, damit die neuen Einstellungen wirksam werden.',
  restartNow: 'Jetzt neu starten',
  restarting: 'Wird neu gestartet…',
  restartFailedManual: 'Gateway-Neustart fehlgeschlagen – starten Sie es manuell neu und prüfen Sie die Gateway-Logs.',
  restartFailedManualDetail:
    'Versuchen Sie den Neustart erneut; wenn er weiterhin fehlschlägt, öffnen Sie die Logs und senden Sie Diagnosedaten.',
  startMessagingGateway: 'Messaging-Gateway starten',
  startingMessagingGateway: 'Messaging-Gateway wird gestartet…',
  gatewayStartFailed: 'Das Messaging-Gateway konnte nicht gestartet werden.'
} satisfies MessagingGatewayTranslations
