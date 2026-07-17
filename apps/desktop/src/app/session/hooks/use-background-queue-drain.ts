import { useStore } from '@nanostores/react'
import { type MutableRefObject, useCallback, useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { resetBrowseState } from '@/store/composer-input-history'
import {
  $queuedPromptsBySession,
  getQueuedPrompts,
  MAX_AUTO_DRAIN_ATTEMPTS,
  type QueuedPromptEntry,
  removeQueuedPrompt,
  shouldAutoDrain
} from '@/store/composer-queue'
import { notify } from '@/store/notifications'
import { $workingSessionIds } from '@/store/session'

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
  const workingSessionIds = useStore($workingSessionIds)
  const submitTextRef = useRef(submitText)
  const drainingSessionIdsRef = useRef(new Set<string>())
  const drainFailuresRef = useRef(new Map<string, number>())
  const retryTimersRef = useRef<number[]>([])
  const [retryTick, setRetryTick] = useState(0)

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
        const failures = (drainFailuresRef.current.get(entry.id) ?? 0) + 1
        drainFailuresRef.current.set(entry.id, failures)

        if (failures >= MAX_AUTO_DRAIN_ATTEMPTS) {
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
          removeQueuedPrompt(sessionKey, liveEntry.id)
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

      if (!entry || (drainFailuresRef.current.get(entry.id) ?? 0) >= MAX_AUTO_DRAIN_ATTEMPTS) {
        continue
      }

      drainSessionQueue(sessionKey, entry)
    }
  }, [drainSessionQueue, enabled, queuedPromptsBySession, retryTick, selectedStoredSessionId, workingSessionIds])
}
