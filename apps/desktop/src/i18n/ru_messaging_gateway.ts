import type { MessagingGatewayTranslations } from './types_messaging_gateway'

export const ruMessagingGateway = {
  hintGatewayStopped: 'Запустите шлюз сообщений здесь для подключения.',
  restartNeeded: 'Сохранено. Перезапустите шлюз сообщений, чтобы применить новые настройки.',
  restartNow: 'Перезапустить',
  restarting: 'Перезапуск…',
  restartFailedManual: 'Не удалось перезапустить шлюз — перезапустите его вручную и проверьте журналы.',
  restartFailedManualDetail:
    'Повторите перезапуск; если он снова не удастся, откройте журналы и отправьте диагностику.',
  startMessagingGateway: 'Запустить шлюз сообщений',
  startingMessagingGateway: 'Запуск шлюза сообщений…',
  gatewayStartFailed: 'Не удалось запустить шлюз сообщений.'
} satisfies MessagingGatewayTranslations
