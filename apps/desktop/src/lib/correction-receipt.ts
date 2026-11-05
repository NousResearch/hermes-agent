import type { SessionRedirectResponse } from '@/app/types'
import { translateNow } from '@/i18n'
import { writeClipboardText } from '@/lib/clipboard'
import { notify } from '@/store/notifications'

/** A failed acknowledgement is not proof that the correction was undelivered. */
export class UnconfirmedCorrectionError extends Error {
  constructor(
    error: unknown,
    public readonly noticeId?: string
  ) {
    super(error instanceof Error ? error.message : 'Correction delivery unconfirmed', { cause: error })
  }
}

export function notifyCorrectionUnconfirmed(text?: string, id?: string): string {
  const show = (copyFailed = false): string =>
    notify({
      ...(id ? { id } : {}),
      kind: 'warning',
      persistent: text !== undefined,
      title: translateNow(copyFailed ? 'common.copyFailed' : 'composer.steerUnconfirmedTitle'),
      message: translateNow(
        text === undefined ? 'composer.steerUnconfirmedBody' : 'composer.steerExternalUnconfirmedBody'
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

/**
 * Where an accepted `queued` correction lands. A backend that names the destination wins:
 * `disposition: 'steer'` joins the live turn at its next tool boundary (a redirect between two
 * model requests degrades to that too), `'queue'` waits for the next turn (the compression
 * guard). An older backend names neither: its steer joins this turn, and its queued redirect is
 * the build window's next turn.
 */
export function queuedCorrectionJoinsLiveTurn(
  result: Pick<SessionRedirectResponse, 'disposition'>,
  mode: 'interrupt' | 'steer'
): boolean {
  if (result.disposition === 'queue') {
    return false
  }

  return result.disposition === 'steer' || mode === 'steer'
}

/** A correction the backend queued for the next turn says so, instead of reading as delivered. */
export function notifyCorrectionQueued(result: Pick<SessionRedirectResponse, 'disposition' | 'reason'>): void {
  if (result.disposition !== 'queue') {
    return
  }

  notify({
    id: 'composer-steer-queued',
    kind: 'info',
    title: translateNow('composer.steerQueuedTitle'),
    message: translateNow(
      result.reason === 'compression' ? 'composer.steerQueuedCompressionBody' : 'composer.steerQueuedBody'
    )
  })
}
