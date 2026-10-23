import type { TranslationOverrides } from './define-locale'

// Temporary (incognito) chat copy, composed by ru.ts.
export const ruTemporaryChat = {
  temporaryChat: {
    title: 'Временный чат',
    description:
      'Временные чаты не отображаются в истории, не используют и не обновляют память и удаляются, когда вы покидаете чат.',
    placeholder: 'Спросите что-нибудь во временном чате',
    use: 'Использовать временный чат',
    exit: 'Выйти из временного чата',
    badge: 'Временный',
    badgeHint: 'Не сохраняется. Завершается, когда вы его закрываете, или после 10 минут простоя, пока вас нет.',
    newChat: 'Новый временный чат',
    unsupportedBackend:
      'Этот бэкенд Hermes пока не поддерживает временные чаты. Обновите его или выключите временный режим, чтобы общаться как обычно.',
    endChat: 'Завершить временный чат',
    endTitle: 'Завершить этот временный чат?',
    endDescription: 'Он будет удалён навсегда — ничего не сохранится.',
    endConfirm: 'Завершить и удалить',
    returnTo: 'Вернуться во временный чат',
    running: 'Ещё отвечает…'
  }
} satisfies TranslationOverrides
