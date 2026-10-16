import type { TranslationOverrides } from './define-locale'
import { ruSessionCommands } from './ru_session_commands'

// Shell notices (remote-display toast, butterbar), spread into ru.ts.
export const ruNotices = {
  remoteDisplayBanner: {
    message: reason =>
      `Включён программный рендеринг — обнаружен удалённый дисплей (${reason}). GPU-ускорение отключено, чтобы избежать мерцания.`
  },
  butterbar: {
    goTo: (index, total) => `Показать уведомление ${index} из ${total}`,
    legal: {
      before: 'Использование Hermes Agent регулируется нашими ',
      terms: 'Условиями обслуживания',
      between: ' и ',
      privacy: 'Политикой конфиденциальности',
      after: '.'
    }
  },
  promptNotices: {
    legacySendUnconfirmed:
      'Этот сервер не смог подтвердить предыдущую отправку этого сообщения, поэтому оно, возможно, уже выполнено. Проверьте беседу, прежде чем отправлять его снова.',
    sessionCommands: ruSessionCommands
  }
} satisfies Pick<TranslationOverrides, 'remoteDisplayBanner' | 'butterbar' | 'promptNotices'>
