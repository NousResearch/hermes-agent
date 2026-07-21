import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { resetBrowseState } from '@/store/composer-input-history'
import {
  $queuedPromptsBySession,
  flushQueuedPromptMutations,
  getQueuedPrompts,
  incrementQueuedPromptAttemptsAtomic,
  markQueuedPromptAcceptedAtomic,
  MAX_AUTO_DRAIN_ATTEMPTS,
  QUEUED_PROMPT_ACCEPTANCE_RETRY_MS,
  queuedPromptAwaitingCompletion,
  removeQueuedPromptByIdAtomic,
  type QueuedPromptEntry,
  refreshQueuedPromptsFromStorage,
  shouldAutoDrain,
  withComposerQueueDrainLease
} from '@/store/composer-queue'
import { notify } from '@/store/notifications'
import { $workingSessionIds } from '@/store/session-states'

import type { SubmitTextOptions, SubmitTextResult } from './use-prompt-actions/utils'

type SubmitQueuedPrompt = (text: string, options?: SubmitTextOptions) => Promise<SubmitTextResult> | SubmitTextResult

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
  const workingSessionIds = useStore($workingSessionIds)
  const submitTextRef = useRef(submitText)
  const drainingSessionIdsRef = useRef(new Set<string>())
  const retryTimersRef = useRef<number[]>([])
  const [retryTick, setRetryTick] = useState(0)

  useEffect(() => {
    submitTextRef.current = submitText
  }, [submitText])

  const scheduleRetry = useCallback((delayMs = BACKGROUND_DRAIN_RETRY_MS) => {
    if (typeof window === 'undefined') {
      return
    }

    const timer = window.setTimeout(() => {
      retryTimersRef.current = retryTimersRef.current.filter(id => id !== timer)
      setRetryTick(tick => tick + 1)
    }, delayMs)

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
        refreshQueuedPromptsFromStorage()

        const current = Object.values($queuedPromptsBySession.get())
          .flat()
          .find(candidate => candidate.id === entry.id)

        if (current && current.attempts >= MAX_AUTO_DRAIN_ATTEMPTS) {
          notify({
            id: `composer-background-queue-stuck-${sessionKey}`,
            kind: 'error',
            title: t.composer.queueStuckTitle,
            message: t.composer.queueStuckBody
          })

          return
        }

        scheduleRetry()
      }

      void Promise.resolve()
        .then(async () => {
          await flushQueuedPromptMutations()

          return withComposerQueueDrainLease(sessionKey, async () => {
            refreshQueuedPromptsFromStorage()
            let liveEntry = getQueuedPrompts(sessionKey).find(candidate => candidate.id === entry.id)

            if (
              !liveEntry ||
              liveEntry.attempts >= MAX_AUTO_DRAIN_ATTEMPTS ||
              queuedPromptAwaitingCompletion(liveEntry)
            ) {
              return true
            }

            liveEntry = (await incrementQueuedPromptAttemptsAtomic(liveEntry.id)) ?? liveEntry

            const runtimeSessionId = runtimeIdByStoredSessionIdRef.current.get(sessionKey) ?? null

            const accepted = await Promise.resolve(
              submitTextRef.current(liveEntry.text, {
                attachments: liveEntry.attachments,
                clientSubmissionId: liveEntry.id,
                fromQueue: true,
                sessionId: runtimeSessionId,
                storedSessionId: sessionKey
              })
            )

            if (accepted === false) {
              return false
            }

            if (typeof accepted === 'object' && accepted.status === 'duplicate') {
              await removeQueuedPromptByIdAtomic(liveEntry.id)
              return true
            }

            if (
              typeof accepted === 'object' &&
              accepted.status !== undefined &&
              accepted.status !== 'queued' &&
              accepted.status !== 'streaming'
            ) {
              return false
            }

            await markQueuedPromptAcceptedAtomic(liveEntry.id)
            scheduleRetry(QUEUED_PROMPT_ACCEPTANCE_RETRY_MS)
            resetBrowseState(runtimeSessionId)

            return true
          })
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
    if (!enabled) {
      return
    }

    const working = new Set(workingSessionIds)

    for (const [sessionKey, entries] of Object.entries(queuedPromptsBySession)) {
      if (
        sessionKey === selectedStoredSessionId ||
        drainingSessionIdsRef.current.has(sessionKey) ||
        !shouldAutoDrain({ isBusy: working.has(sessionKey), queueLength: entries.length })
      ) {
        continue
      }

      const entry = entries[0]

      if (!entry || entry.attempts >= MAX_AUTO_DRAIN_ATTEMPTS || queuedPromptAwaitingCompletion(entry)) {
        continue
      }

      drainSessionQueue(sessionKey, entry)
    }
  }, [drainSessionQueue, enabled, queuedPromptsBySession, retryTick, selectedStoredSessionId, workingSessionIds])
}
