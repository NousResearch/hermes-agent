import { useEffect } from 'react'

import { translateNow } from '@/i18n'
import { activeGateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'

import { holdQueuedPrompt } from '../discard-lost-prompt'

/** The owner's hold lapses after 120 s; an open edit renews it well before that. */
const RENEW_MS = 60_000

/** Refusals that mean the row is no longer queued (it started, or a save replaced it): the save
 *  path answers those itself, so the hold has nothing left to protect. */
const ROW_GONE = new Set(['admission_started', 'not_found'])

function refusalReason(error: unknown): string {
  const data = error && typeof error === 'object' ? (error as { data?: unknown }).data : undefined
  const reason = data && typeof data === 'object' ? (data as { reason?: unknown }).reason : undefined

  return typeof reason === 'string' ? reason : ''
}

/**
 * While a server-queued row is open in the composer, ask its owner to hold it so the row cannot
 * start mid-edit. The hold is renewed while the edit stays open and released when it closes; a
 * saved edit replaces the row, so the release then finds nothing to release. Each open edit holds
 * under its own token, so closing one window's edit never releases another's. A hold that fails
 * for any other reason is surfaced once: the row is then unprotected and may start before Save.
 */
export function useServerEditHold(
  sessionId: string | null | undefined,
  queueSessionKey: string | null | undefined,
  admissionId: string | null
): void {
  useEffect(() => {
    const gateway = activeGateway()

    if (!gateway || !queueSessionKey || !admissionId) {
      return
    }

    const request = gateway.request.bind(gateway)
    const holder = crypto.randomUUID()
    let warned = false

    const hold = (held: boolean) =>
      void holdQueuedPrompt(sessionId, queueSessionKey, admissionId, held, request, holder).catch((error: unknown) => {
        if (!held || warned || ROW_GONE.has(refusalReason(error))) {
          return
        }

        warned = true
        notifyError(error, translateNow('composer.queueFair.holdFailed'))
      })

    hold(true)
    const timer = window.setInterval(() => hold(true), RENEW_MS)

    return () => {
      window.clearInterval(timer)
      hold(false)
    }
  }, [admissionId, queueSessionKey, sessionId])
}
