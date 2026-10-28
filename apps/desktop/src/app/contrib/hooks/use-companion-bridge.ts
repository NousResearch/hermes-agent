import { useEffect, useRef } from 'react'

import { setCompanionActivity } from '@/store/companion'
import { setCompanionScale } from '@/store/companion-gallery'
import { setCompanionOverlayOpenAppHandler, setCompanionOverlayScaleHandler, setCompanionOverlaySubmitHandler } from '@/store/companion-overlay'
import { $sessions } from '@/store/session'
import { $attentionSessionIds, $workingSessionIds } from '@/store/session-states'
import { isAuxiliaryWindow } from '@/store/windows'

import type { GatewayRequester } from '../types'

/**
 * Mirror "any open session is running the agent" into the companion's `busy` flag.
 *
 * `toolRunning` / `reasoning` are set only inside the per-session message-stream
 * hook (`use-message-stream/gateway-event.ts`), which is mounted for the
 * foreground session alone — so a background session running tools left the companion
 * stuck in `idle` (#84282). `$workingSessionIds` is a store-level computed over
 * every session's authoritative `busy` state (the same cross-session source
 * `$attentionSessionIds` provides for the `waiting` pose), so it stays live
 * regardless of which session is rendered. `deriveCompanionState` maps `busy` to the
 * `run` pose, so any working session — foreground or background — now animates
 * the companion, while the foreground stream's finer `toolRunning` / `reasoning`
 * signals still refine run-vs-review for the visible session.
 */
export function syncCompanionBusyFromSessions(): void {
  setCompanionActivity({ busy: $workingSessionIds.get().length > 0 })
}

interface CompanionBridgeParams {
  requestGateway: GatewayRequester
  resumeSession: (sessionId: string) => Promise<unknown> | unknown
  submitText: (text: string) => Promise<unknown> | unknown
}

/**
 * Wires the popped-out companion overlay back into the app: submit a prompt, resize,
 * and open the most-recent thread, plus mirroring "a session is awaiting the
 * user" into the companion's pose. Handlers register ONCE through refs tracking the
 * latest callbacks — re-registering on identity churn leaves a nulled-handler
 * window that can drop a submit. Primary window only.
 */
export function useCompanionBridge({ requestGateway, resumeSession, submitText }: CompanionBridgeParams): void {
  const submitTextRef = useRef(submitText)
  submitTextRef.current = submitText
  const resumeSessionRef = useRef(resumeSession)
  resumeSessionRef.current = resumeSession
  const requestGatewayRef = useRef(requestGateway)
  requestGatewayRef.current = requestGateway

  useEffect(() => {
    if (isAuxiliaryWindow()) {
      return
    }

    setCompanionOverlaySubmitHandler(text => void submitTextRef.current(text))
    // Alt+wheel resize from the popped-out companion — persist through this window's
    // gateway (the overlay has none) so it survives restart.
    setCompanionOverlayScaleHandler(scale => setCompanionScale(requestGatewayRef.current, scale))
    // Mail icon: $sessions is most-recent-first; the companion is global, so "most
    // recent" is the right target.
    setCompanionOverlayOpenAppHandler(() => {
      const recent = $sessions.get()[0]

      if (recent?.id) {
        void resumeSessionRef.current(recent.id)
      }
    })

    return () => {
      setCompanionOverlaySubmitHandler(null)
      setCompanionOverlayOpenAppHandler(null)
      setCompanionOverlayScaleHandler(null)
    }
  }, [])

  // Mirror "a session is blocked on the user" (clarify/approval) into the companion's
  // awaitingInput flag so it shows the `waiting` pose.
  useEffect(() => {
    const sync = () => setCompanionActivity({ awaitingInput: $attentionSessionIds.get().length > 0 })

    sync()

    return $attentionSessionIds.listen(sync)
  }, [])

  // Mirror "any session is running the agent" into the companion's busy flag so the
  // companion reacts to background sessions too, not just the foreground one (#84282).
  useEffect(() => {
    syncCompanionBusyFromSessions()

    return $workingSessionIds.listen(syncCompanionBusyFromSessions)
  }, [])
}
