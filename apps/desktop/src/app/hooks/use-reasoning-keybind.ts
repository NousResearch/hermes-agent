import { useRef } from 'react'

import { PRIMARY_SESSION_VIEW } from '@/app/chat/session-view'
import { translateNow } from '@/i18n/runtime'
import { stepReasoningEffort, writeSessionReasoningEffort } from '@/lib/reasoning-step'
import {
  getComposerReasoningEffort,
  getComposerReasoningGeneration,
  getComposerReasoningScope,
  setComposerReasoningEffort
} from '@/store/composer-reasoning-effort'
import { notifyError } from '@/store/notifications'
import {
  $activeSessionId,
  $defaultReasoningEffort,
  markComposerSelectionManual,
  setCurrentReasoningEffort
} from '@/store/session'
import { sessionTileDelegate } from '@/store/session-states'

interface ReasoningWrite {
  sessionId: string
  scope: string | null
  generation: number
  confirmed: { effort: string; pick: string }
  pending: Promise<void> | null
}

type RequestGateway = <T>(method: string, params?: Record<string, unknown>) => Promise<T>

function publishEffort(sessionId: string | null, effort: string, pick: string): void {
  setComposerReasoningEffort(pick)
  setCurrentReasoningEffort(effort)

  if (sessionId) {
    sessionTileDelegate()?.updateSession(sessionId, state => ({
      ...state,
      reasoningEffort: effort,
      reasoningEffortWire: ''
    }))
  }
}

export function useReasoningKeybind(requestGateway: RequestGateway): (direction: 1 | -1) => void {
  const writes = useRef<ReasoningWrite | null>(null)

  return direction => {
    const sessionId = $activeSessionId.get()
    const scope = getComposerReasoningScope()
    const previousGeneration = getComposerReasoningGeneration()
    const pick = getComposerReasoningEffort()
    const effort = sessionId ? PRIMARY_SESSION_VIEW.$reasoningEffort.get() : pick
    const next = stepReasoningEffort(effort, direction, $defaultReasoningEffort.get() || undefined)

    if (next === effort.trim().toLowerCase()) {
      return
    }

    markComposerSelectionManual()
    publishEffort(sessionId, next, next)
    const stamp = getComposerReasoningGeneration()

    // A draft edit must never fall through to persistent profile config.
    if (!sessionId) {
      return
    }

    let write = writes.current

    if (
      !write?.pending ||
      write.sessionId !== sessionId ||
      write.scope !== scope ||
      write.generation !== previousGeneration
    ) {
      write = { sessionId, scope, generation: stamp, confirmed: { effort, pick }, pending: null }
      writes.current = write
    }

    const chain = write
    chain.generation = stamp

    const isCurrent = (generation = stamp) =>
      getComposerReasoningGeneration() === generation &&
      $activeSessionId.get() === sessionId &&
      getComposerReasoningScope() === scope

    // Keep the last confirmed selection across overlapping requests. A failed
    // newer edit must not roll back to another unconfirmed optimistic pick.
    const pending = (chain.pending ?? Promise.resolve()).then(async () => {
      try {
        // A menu edit revokes unsent keyboard writes, not just their eventual paint.
        if (!isCurrent(chain.generation)) {
          return
        }

        const value = await writeSessionReasoningEffort(requestGateway, sessionId, next)
        chain.confirmed = { effort: value, pick: value }

        if (isCurrent()) {
          publishEffort(sessionId, value, value)
        }
      } catch (error) {
        if (isCurrent()) {
          publishEffort(sessionId, chain.confirmed.effort, chain.confirmed.pick)
          notifyError(error, translateNow('shell.modelOptions.updateFailed'))
        }
      } finally {
        if (chain.pending === pending) {
          chain.pending = null
        }
      }
    })

    chain.pending = pending
  }
}
