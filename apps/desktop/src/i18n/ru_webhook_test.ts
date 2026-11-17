import type { WebhookTestCopy } from './types_webhook_test'

export const ruWebhookTest: Partial<WebhookTestCopy> = {
  title: 'Проверить этот вебхук',
  hint: 'Вставьте в терминал, чтобы отправить одно подписанное событие. У каждого запуска свой идентификатор доставки.',
  placeholderHint: 'Замените заполнитель секретом, показанным при создании подписки, или используйте «Отправить тест».',
  copyCommand: 'Скопировать команду',
  send: 'Отправить тест',
  sending: 'Отправка…',
  accepted: (status, deliveryId) => `Тест доставлен (${status}) · ${deliveryId}`,
  rejected: (status, detail) => `Получатель ответил ${status}: ${detail}`,
  ignored: 'Доставлено, но подписка проигнорировала его (её правила событий или фильтров отбрасывают тестовое событие)',
  failed: 'Не удалось отправить тест'
}
