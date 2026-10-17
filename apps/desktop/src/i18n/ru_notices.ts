import type { TranslationOverrides } from './define-locale'
import { ruSessionCommands } from './ru_session_commands'

// Shell notices (remote-display toast, butterbar, keep-awake offer), spread into ru.ts.
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
    mcpReloadDeferred:
      'Чат занят: изменение MCP применится, когда завершится текущий ход и MCP будет перезагружен снова, или при повторном открытии чата',
    mcpReloadNotInChat: server => `${server} не включён в этом чате`,
    mcpReloadUnavailable: servers => `Не удалось подключиться: ${servers}`,
    sessionCommands: ruSessionCommands
  },
  keepAwakeOffer: {
    title: 'Компьютер ушёл в сон, пока Hermes работал',
    message:
      'Hermes может не давать компьютеру засыпать, пока выполняется задача. Экран по-прежнему выключается, а от батареи компьютер всё равно может уснуть.',
    action: 'Не засыпать во время работы'
  }
} satisfies Pick<TranslationOverrides, 'remoteDisplayBanner' | 'butterbar' | 'promptNotices' | 'keepAwakeOffer'>
