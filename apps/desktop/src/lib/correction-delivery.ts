import { translateNow } from '@/i18n'
import { writeClipboardText } from '@/lib/clipboard'
import { notify } from '@/store/notifications'

/** An absent/lost acknowledgement cannot authorize a new turn. */
export class UnconfirmedCorrectionError extends Error {
  constructor(
    error: unknown,
    public readonly noticeId?: string
  ) {
    super(error instanceof Error ? error.message : 'Correction delivery unconfirmed', { cause: error })
  }
}

/** External requests have no draft: retain their only copy without automatically replaying it. */
export function notifyCorrectionRecovery(text?: string, notSent = false, id?: string): string {
  const show = (copyFailed = false): string =>
    notify({
      ...(id ? { id } : {}),
      kind: 'warning',
      persistent: text !== undefined,
      title: translateNow(
        copyFailed ? 'common.copyFailed' : notSent ? 'composer.steerNotSentTitle' : 'composer.steerUnconfirmedTitle'
      ),
      message: translateNow(
        notSent
          ? 'composer.steerExternalNotSentBody'
          : text === undefined
            ? 'composer.steerUnconfirmedBody'
            : 'composer.steerExternalUnconfirmedBody'
      ),
      ...(text === undefined
        ? {}
        : {
            action: {
              label: translateNow('common.copy'),
              onClick: () => {
                void writeClipboardText(text).catch(() => show(true))
              }
            }
          })
    })

  return show()
}
