// Arbitration for the `speak:<messageId>` ambient cue. Pure — no Electron
// import — so the decision is unit-testable without a window.

/** A speech claim stands for the life of one reply, not one event tick. */
export const SPEECH_CLAIM_TTL_MS = 60_000

const SPEECH_CUE_PREFIX = 'speak:'

/** `speak:<messageId>` — the read-aloud cue, keyed by backend message id. */
export function isSpeechCue(key: string): boolean {
  return key.startsWith(SPEECH_CUE_PREFIX)
}

export interface SpeechClaimContext {
  /** The claim came from the HUD's own renderer. */
  fromHud: boolean
  /** A HUD window is up right now. */
  hudOpen: boolean
  /** The claimer is the app window this HUD hid on its way up, still hidden. */
  senderDisplacedByHud: boolean
}

/**
 * True when an open HUD outranks this claim and the caller must stay quiet.
 * Only speech is arbitrated: the turn-end sound is one instant beep the
 * existing first-caller-wins collapse already handles.
 */
export function hudOutranksSpeechClaim(
  key: string,
  { fromHud, hudOpen, senderDisplacedByHud }: SpeechClaimContext
): boolean {
  if (!hudOpen || fromHud || !senderDisplacedByHud) {
    return false
  }

  return isSpeechCue(key)
}
