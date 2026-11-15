import { useEffect, useRef } from 'react'

import { setPetActivity } from '@/store/pet'
import { setPetScale } from '@/store/pet-gallery'
import { setPetOverlayOpenAppHandler, setPetOverlayScaleHandler, setPetOverlaySubmitHandler } from '@/store/pet-overlay'
import { $sessions } from '@/store/session'
import { $attentionSessionIds, $workingSessionIds } from '@/store/session-states'
import { isAuxiliaryWindow } from '@/store/windows'

import type { GatewayRequester } from '../types'

/**
 * Mirror "any open session is running the agent" into the pet's `busy` flag.
 *
 * `toolRunning` / `reasoning` are set only inside the per-session message-stream
 * hook (`use-message-stream/gateway-event.ts`), which is mounted for the
 * foreground session alone — so a background session running tools left the pet
 * stuck in `idle` (#84282). `$workingSessionIds` is a store-level computed over
 * every session's authoritative `busy` state (the same cross-session source
 * `$attentionSessionIds` provides for the `waiting` pose), so it stays live
 * regardless of which session is rendered. `derivePetState` maps `busy` to the
 * `run` pose, so any working session — foreground or background — now animates
 * the pet, while the foreground stream's finer `toolRunning` / `reasoning`
 * signals still refine run-vs-review for the visible session.
 */
export function syncPetBusyFromSessions(): void {
  setPetActivity({ busy: $workingSessionIds.get().length > 0 })
}

interface PetBridgeParams {
  requestGateway: GatewayRequester
  resumeSession: (sessionId: string) => Promise<unknown> | unknown
  submitText: (text: string) => Promise<unknown> | unknown
}

/**
 * Wires the popped-out pet overlay back into the app: submit a prompt, resize,
 * and open the most-recent thread, plus mirroring "a session is awaiting the
 * user" into the pet's pose. Handlers register ONCE through refs tracking the
 * latest callbacks — re-registering on identity churn leaves a nulled-handler
 * window that can drop a submit. Primary window only.
 */
export function usePetBridge({ requestGateway, resumeSession, submitText }: PetBridgeParams): void {
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

    setPetOverlaySubmitHandler(text => void submitTextRef.current(text))
    // Alt+wheel resize from the popped-out pet — persist through this window's
    // gateway (the overlay has none) so it survives restart.
    setPetOverlayScaleHandler(scale => setPetScale(requestGatewayRef.current, scale))
    // Mail icon: $sessions is most-recent-first; the pet is global, so "most
    // recent" is the right target.
    setPetOverlayOpenAppHandler(() => {
      const recent = $sessions.get()[0]

      if (recent?.id) {
        void resumeSessionRef.current(recent.id)
      }
    })

    return () => {
      setPetOverlaySubmitHandler(null)
      setPetOverlayOpenAppHandler(null)
      setPetOverlayScaleHandler(null)
    }
  }, [])

  // Mirror "a session is blocked on the user" (clarify/approval) into the pet's
  // awaitingInput flag so it shows the `waiting` pose.
  useEffect(() => {
    const sync = () => setPetActivity({ awaitingInput: $attentionSessionIds.get().length > 0 })

    sync()

    return $attentionSessionIds.listen(sync)
  }, [])

  // Mirror "any session is running the agent" into the pet's busy flag so the
  // pet reacts to background sessions too, not just the foreground one (#84282).
  useEffect(() => {
    syncPetBusyFromSessions()

    return $workingSessionIds.listen(syncPetBusyFromSessions)
  }, [])
}
