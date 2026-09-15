// Orb-state hook: project the desktop's session state onto `OrbState`.
//
// Reads only the session-view stores — deliberately NOT the assistant-ui
// message runtime, so it works anywhere the orb renders (thread status rows,
// the composer jewel, session tiles), including surfaces rendered without a
// message scope. The session mirror (`view.$messages`) carries the tail
// parts with per-delta updates, so the state is as live as the transcript.

import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { sessionCompacting } from '@/store/compaction'
import { sessionAwaitingInput } from '@/store/prompts'
import { sessionProviderWait } from '@/store/provider-wait'
import { type DraftingTool, sessionDraftingTool } from '@/store/tool-drafting'

import { type OrbPartLike, type OrbState, orbTailPhase, resolveOrbState } from './orb-state'

/** How long the `complete` settle flash shows before the orb goes idle. */
const COMPLETE_HOLD_MS = 2500

/**
 * Flip `justCompleted` on for COMPLETE_HOLD_MS after a running turn settles
 * successfully, then let it fall back to idle. The transition is detected
 * with a render-phase state update (no ref mirror — refs synced from reactive
 * values lag one render).
 */
function useJustCompleted(running: boolean, failed: boolean): boolean {
  const [flash, setFlash] = useState(false)
  const [prevRunning, setPrevRunning] = useState(running)

  if (prevRunning !== running) {
    setPrevRunning(running)
    setFlash(prevRunning && !running && !failed)
  }

  useEffect(() => {
    if (!flash) {
      return
    }

    const id = window.setTimeout(() => setFlash(false), COMPLETE_HOLD_MS)

    return () => window.clearTimeout(id)
  }, [flash])

  return flash
}

/**
 * The orb state for a session, from the session-view stores alone. The
 * thread's thinking dot and the composer jewel both read this, so they can
 * never disagree about what the assistant is doing.
 */
export function useOrbState(): OrbState {
  const view = useSessionView()
  const sessionId = useStore(view.$runtimeId)
  const busy = useStore(view.$busy)
  const messages = useStore(view.$messages)
  const awaitingInput = useStore(useMemo(() => sessionAwaitingInput(sessionId), [sessionId]))
  const compacting = useStore(useMemo(() => sessionCompacting(sessionId), [sessionId]))
  const providerWait = useStore(useMemo(() => sessionProviderWait(sessionId), [sessionId]))
  const drafting = useStore(useMemo(() => sessionDraftingTool(sessionId), [sessionId]))

  const lastAssistant = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i]

      if (message.role === 'assistant' && !message.hidden) {
        return message
      }
    }

    return undefined
  }, [messages])

  const lastError = (lastAssistant as { error?: unknown } | undefined)?.error
  const messageError = lastError !== undefined && lastError !== ''
  const running = busy || lastAssistant?.pending === true
  const justCompleted = useJustCompleted(running, messageError)

  const tailPhase = useMemo(() => orbTailPhase((lastAssistant?.parts ?? []) as readonly OrbPartLike[]), [lastAssistant])

  return resolveOrbState({
    awaitingInput,
    busy,
    compacting,
    draftingTool: (drafting as DraftingTool | null) !== null,
    justCompleted,
    messageError,
    providerWait: providerWait !== '',
    statusReason: messageError ? 'error' : undefined,
    statusType: messageError ? 'incomplete' : running ? 'running' : undefined,
    tailPhase
  })
}
