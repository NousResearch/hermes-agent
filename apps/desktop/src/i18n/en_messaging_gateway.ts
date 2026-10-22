import type { MessagingGatewayTranslations } from './types_messaging_gateway'

export const enMessagingGateway = {
  hintGatewayStopped: 'Start the messaging gateway here to connect.',
  restartNeeded: 'Saved. Restart the messaging gateway so the new settings take effect.',
  restartNow: 'Restart now',
  restarting: 'Restarting…',
  restartFailedManual: "Hermes couldn't restart to apply your messaging settings",
  restartFailedManualDetail: 'Try Restart again; if it still fails, open the logs and send diagnostics.',
  startMessagingGateway: 'Start messaging gateway',
  startingMessagingGateway: 'Starting messaging gateway…',
  gatewayStartFailed: 'Messaging gateway start failed.'
} satisfies MessagingGatewayTranslations
