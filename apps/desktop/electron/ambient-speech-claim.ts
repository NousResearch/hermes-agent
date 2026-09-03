// HUD-scoped ownership for spoken replies (#99717). The ambient-cue claim IPC
// backs both turn-end sounds and spoken replies with the same 1s event deduper,
// which fits near-simultaneous cues but not message-level speech: a long reply
// can play for minutes, and once the dedupe window expires a second renderer's
// delayed claim wins and replays the same reply. While a HUD window is live it
// is the surface the user is looking at, so speech ownership is pinned to it by
// sender identity instead of wall-clock racing.

export interface AmbientSpeechClaim {
  key: string
  senderId: number
  hudSenderId: number | null
}

/**
 * Decide `speak:*` cue ownership when a HUD window is live: true/false pins the
 * claim to (or away from) the HUD renderer. Returns null when the generic
 * first-caller-wins deduper should decide instead — non-speech cues, or no live
 * HUD — so `sound:*` cues and normal multi-window behavior are unchanged.
 */
export function hudScopedSpeechOwnership({
  key,
  senderId,
  hudSenderId
}: AmbientSpeechClaim): boolean | null {
  if (!key.startsWith('speak:') || hudSenderId === null) {
    return null
  }

  return senderId === hudSenderId
}
