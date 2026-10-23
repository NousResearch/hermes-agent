import type { Translations } from './types'

type ThreadCopy = Translations['assistant']['thread']

// The error card's per-code title and body (lib/error-surface-copy.ts), keyed by error_surface.code.
export const ruErrorCodes: Partial<ThreadCopy['errorCodes']> = {
  cron_session_read_only: {
    title: 'Этот чат относится к запланированной задаче',
    body: 'Он только выполняет эту задачу, поэтому ваше сообщение не было выполнено. Начните новый чат, чтобы отправить его.'
  },
  workspace_unavailable: {
    title: 'Папка этого чата пропала',
    body: 'Папка, в которой работает этот чат, была перемещена или удалена, поэтому ваше сообщение не было выполнено. Создайте папку заново или назначьте чату другую через «Переместить в проект», затем повторите попытку.'
  },
  usage_window: {
    title: 'Включённый объём исчерпан',
    body: 'Включённый объём вашего плана исчерпан до сброса. Пополните кредиты в /topup, чтобы продолжить.'
  }
}
