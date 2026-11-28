import type { BusyInputMode } from '@/store/busy-input-mode'

export type ComposerBusyAction = 'interrupt' | 'steer' | 'queue' | 'stop'

export interface BusySubmitInputs {
  attachmentCount: number
  blockingPrompt: boolean
  busy: boolean
  busyInputMode: BusyInputMode | null
  compacting: boolean
  hasComposerPayload: boolean
  hasSteerHandler: boolean
  isSteerableText: boolean
}

export interface BusySubmitState {
  busyAction: ComposerBusyAction
  /** The draft can end the live turn and run as the next one — any payload, attachments included. */
  canInterrupt: boolean
  canSteer: boolean
  canSubmit: boolean
}

/** The Send button's gate plus the busy-turn routing (steer / queue / stop). */
export function deriveBusySubmitState({
  attachmentCount,
  blockingPrompt,
  busy,
  busyInputMode,
  compacting,
  hasComposerPayload,
  hasSteerHandler,
  isSteerableText
}: BusySubmitInputs): BusySubmitState {
  const canSubmit =
    (busy || hasComposerPayload) &&
    !(busy && isSteerableText && attachmentCount === 0 && !compacting && !blockingPrompt && busyInputMode === null)

  // Steer only makes sense mid-turn, text-only (the gateway can't carry images
  // into a tool result) and never for a slash command (those execute inline).
  // A blocking prompt (approval/sudo/secret) also rules it out: the tool batch
  // is parked on the user, so a steer can't reach the model — text queues.
  // Compaction does not: the gateway holds the correction until it finishes.
  const canSteer = busy && !blockingPrompt && hasSteerHandler && attachmentCount === 0 && isSteerableText

  // Interrupt is not steer. A redirect has to inject the correction into the
  // running turn, which is why it cannot carry an image; an interrupt ends the
  // turn and lets the draft run as the next one, so the image travels the
  // ordinary attachment path instead. Any payload can interrupt, but only where
  // stopping the turn is legitimate: queue mode is the user asking to wait, and
  // a parked prompt or an in-flight compaction must not be killed.
  //
  // `isSteerableText || attachmentCount > 0` is "there is something to send that
  // is not a slash command" — a bare `/command` still queues, as before.
  const canInterrupt =
    busy &&
    !blockingPrompt &&
    !compacting &&
    hasComposerPayload &&
    busyInputMode !== 'queue' &&
    busyInputMode !== null &&
    (isSteerableText || attachmentCount > 0)

  // Ordinary busy Send follows backend policy; a draft carrying attachments
  // interrupts instead of silently waiting for the turn to end; an empty
  // composer stops.
  const busyAction: ComposerBusyAction = canSteer
    ? (busyInputMode ?? 'interrupt')
    : canInterrupt
      ? 'interrupt'
      : compacting || hasComposerPayload
        ? 'queue'
        : 'stop'

  return { busyAction, canInterrupt, canSteer, canSubmit }
}
