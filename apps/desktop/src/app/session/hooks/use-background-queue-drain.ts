import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { resetBrowseState } from '@/store/composer-input-history'
import {
  $parkedQueueSessions,
  $queuedPromptsBySession,
  getQueuedPrompts,
  MAX_AUTO_DRAIN_ATTEMPTS,
  noteQueuedPromptDrainFailure,
  type QueuedPromptEntry,
  removeQueuedPrompt,
  shouldAutoDrain
} from '@/store/composer-queue'
import { notify } from '@/store/notifications'
import {
  $sessionProfilesTruncated,
  $sessions,
  $sessionsLoadError,
  $sessionsLoading,
  getSessionOwnerHints,
  idsShareLineage,
  sessionMatchesStoredId
} from '@/store/session'
import { $workingSessionIds } from '@/store/session-states'

import type { SubmitTextOptions } from './use-prompt-actions/utils'

type SubmitQueuedPrompt = (text: string, options?: SubmitTextOptions) => Promise<boolean> | boolean

interface BackgroundQueueDrainOptions {
  enabled: boolean
  runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>>
  selectedStoredSessionId: string | null
  submitText: SubmitQueuedPrompt
}

const BACKGROUND_DRAIN_RETRY_MS = 750

/**
 * Drain queued prompts for sessions that are not currently rendered by ChatBar.
 *
 * The visible ChatBar owns the interactive queue panel for the selected session.
 * Without this background drain, a prompt queued in Session A can sit forever
 * after the user switches to Session B: the only auto-drain effect lives inside
 * the mounted ChatBar, so Session A's queue is not observed when A is offscreen.
 */
export function useBackgroundQueueDrain({
  enabled,
  runtimeIdByStoredSessionIdRef,
  selectedStoredSessionId,
  submitText
}: BackgroundQueueDrainOptions) {
  const { t } = useI18n()
  const queuedPromptsBySession = useStore($queuedPromptsBySession)
  const parkedQueueSessions = useStore($parkedQueueSessions)
  const sessionsLoading = useStore($sessionsLoading)
  const workingSessionIds = useStore($workingSessionIds)
  const submitTextRef = useRef(submitText)
  const drainingSessionIdsRef = useRef(new Set<string>())
  const drainFailuresRef = useRef(new Map<string, number>())
  const retryTimersRef = useRef<number[]>([])
  const [retryTick, setRetryTick] = useState(0)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    submitTextRef.current = submitText
  }, [submitText])

  const scheduleRetry = useCallback(() => {
    if (typeof window === 'undefined') {
      return
    }

    const timer = window.setTimeout(() => {
      retryTimersRef.current = retryTimersRef.current.filter(id => id !== timer)
      setRetryTick(tick => tick + 1)
    }, BACKGROUND_DRAIN_RETRY_MS)

    retryTimersRef.current.push(timer)
  }, [])

  useEffect(
    () => () => {
      for (const timer of retryTimersRef.current) {
        window.clearTimeout(timer)
      }

      retryTimersRef.current = []
    },
    []
  )

  const drainSessionQueue = useCallback(
    (sessionKey: string, entry: QueuedPromptEntry) => {
      if (drainingSessionIdsRef.current.has(sessionKey)) {
        return
      }

      drainingSessionIdsRef.current.add(sessionKey)

      const onFail = () => {
        const failures = (drainFailuresRef.current.get(entry.id) ?? entry.drainFailures ?? 0) + 1
        drainFailuresRef.current.set(entry.id, failures)
        // Persist the budget with the queue: a restart must not replay four
        // more rejections (and the exhaustion notice) against a session that
        // is exactly as dead as it was when the process exited (#98015).
        noteQueuedPromptDrainFailure(sessionKey, entry.id)

        if (failures >= MAX_AUTO_DRAIN_ATTEMPTS) {
          // The session rejected every drain attempt. Discovery has settled
          // (the effect gates on it), so the loaded list plus owner hints are
          // authoritative — but only when they can actually PROVE absence.
          // "Maybe" must never mean delete:
          // - getSessionOwnerHint is undefined both for "no route" AND for
          //   "two or more routes" (a cloud gateway plus a local backend);
          //   the plural accessor keeps those apart, and ≥1 route is alive.
          // - $sessions is one PAGE of the sidebar list. A session that fell
          //   off the loaded window ($sessionProfilesTruncated) is unknown
          //   by row and hint, not gone.
          // Only a session no row, no hint AND a complete, untruncated list
          // answer to — by id or lineage — is gone from this backend (deleted
          // from another surface, or its stored resume refuses permanently).
          // Owner hints count: a hidden bot chat never occupies the recents
          // list, yet its queue is exactly the one worth preserving. A truly
          // gone session's queued prompt can never send; drop it and say so
          // quietly. An unprovable one keeps its entry for a manual send.
          const sessionKnown =
            $sessions.get().some(session => sessionMatchesStoredId(session, sessionKey)) ||
            getSessionOwnerHints(sessionKey).length > 0

          const listIncomplete =
            $sessionsLoadError.get() || Object.values($sessionProfilesTruncated.get()).some(Boolean)

          if (!sessionKnown && !listIncomplete) {
            removeQueuedPrompt(sessionKey, entry.id)
            notify({
              id: `composer-background-queue-stuck-${sessionKey}`,
              kind: 'info',
              title: t.composer.queueDroppedTitle,
              message: t.composer.queueDroppedBody
            })

            return
          }

          // The conversation still exists — the runtime just would not come
          // back (backend restarting, resume refusing). Keep the entry: it
          // is real data the user can still send from the queue panel, and a
          // manual send clears the retry budget. Downgrade the notice from
          // the old error banner: "not sent, still queued, try again" is
          // accurate, "message not sent" as an ERROR read as data loss.
          notify({
            id: `composer-background-queue-stuck-${sessionKey}`,
            kind: 'info',
            title: t.composer.queueStuckTitle,
            message: t.composer.queueStuckBody
          })

          return
        }

        scheduleRetry()
      }

      void Promise.resolve()
        .then(async () => {
          const liveEntry = getQueuedPrompts(sessionKey).find(candidate => candidate.id === entry.id)

          if (!liveEntry) {
            return true
          }

          const runtimeSessionId = runtimeIdByStoredSessionIdRef.current.get(sessionKey) ?? null

          const accepted = await Promise.resolve(
            submitTextRef.current(liveEntry.text, {
              attachments: liveEntry.attachments,
              fromQueue: true,
              sessionId: runtimeSessionId,
              storedSessionId: sessionKey
            })
          )

          if (accepted === false) {
            return false
          }

          drainFailuresRef.current.delete(liveEntry.id)
          // Submit owns blob: previews after a successful drain handoff.
          removeQueuedPrompt(sessionKey, liveEntry.id, { retainPreviewUrls: true })
          resetBrowseState(runtimeSessionId)

          return true
        })
        .then(accepted => {
          if (!accepted) {
            onFail()
          }
        })
        .catch(onFail)
        .finally(() => {
          drainingSessionIdsRef.current.delete(sessionKey)
        })
    },
    [runtimeIdByStoredSessionIdRef, scheduleRetry, t]
  )

  useEffect(() => {
    // Preserve the retry budget while session discovery runs at boot, on a
    // gateway/profile switch, or during a refresh over an empty list.
    // Once discovery settles, submitText can resume by stored id.
    if (!enabled || sessionsLoading) {
      return
    }

    // Queue keys prefer the lineage root (resolveComposerSessionKey) while
    // $workingSessionIds / selection may hold the compression tip. Strict
    // equality then mis-classifies a busy or selected chat as idle/offscreen.
    const sessions = $sessions.get()
    const working = [...workingSessionIds]

    for (const [sessionKey, entries] of Object.entries(queuedPromptsBySession)) {
      const isSelected =
        Boolean(selectedStoredSessionId) && idsShareLineage(sessionKey, selectedStoredSessionId!, sessions)

      const isBusy = working.some(workingId => idsShareLineage(sessionKey, workingId, sessions))

      if (
        isSelected ||
        drainingSessionIdsRef.current.has(sessionKey) ||
        !shouldAutoDrain({
          isBusy,
          parked: Boolean(parkedQueueSessions[sessionKey]),
          queueLength: entries.length
        })
      ) {
        continue
      }

      const entry = entries[0]

      if (!entry || (drainFailuresRef.current.get(entry.id) ?? entry.drainFailures ?? 0) >= MAX_AUTO_DRAIN_ATTEMPTS) {
        continue
      }

      drainSessionQueue(sessionKey, entry)
    }
  }, [
    drainSessionQueue,
    enabled,
    parkedQueueSessions,
    queuedPromptsBySession,
    retryTick,
    selectedStoredSessionId,
    sessionsLoading,
    workingSessionIds
  ])
}
