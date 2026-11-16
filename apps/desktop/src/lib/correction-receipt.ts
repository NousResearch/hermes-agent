import type { SessionRedirectResponse } from '@/app/types'
import { translateNow } from '@/i18n'
import { notify } from '@/store/notifications'

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
    message: translateNow(result.reason === 'compression' ? 'composer.steerQueuedCompressionBody' : 'composer.steerQueuedBody')
  })
}
