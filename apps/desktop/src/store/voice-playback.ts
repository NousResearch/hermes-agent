import { atom } from 'nanostores'

export type VoicePlaybackSource = 'read-aloud' | 'voice-conversation'
export type VoicePlaybackStatus = 'idle' | 'preparing' | 'speaking'

export interface VoicePlaybackState {
  audioElement: HTMLAudioElement | null
  messageId: string | null
  sequence: number
  /** Session that owns this clip. One renderer can have several sessions on
   *  screen at once (tiles), and there is one speaker, so playback stays a
   *  single slot — but every surface must know WHOSE slot it is before it
   *  reacts. A stop keeps the stopped session's id on the idle state so a
   *  session can tell its own clip was cut. */
  sessionId: string | null
  source: VoicePlaybackSource | null
  status: VoicePlaybackStatus
}

export const $voicePlayback = atom<VoicePlaybackState>({
  audioElement: null,
  messageId: null,
  sequence: 0,
  sessionId: null,
  source: null,
  status: 'idle'
})

/** Normalize a session id for comparison: a fresh chat passes `''`, the
 *  playback state stores `null`, and both mean "no session yet". */
export function voicePlaybackSessionKey(sessionId: string | null | undefined): string | null {
  return sessionId || null
}

/** Does `playback` belong to `sessionId`? One renderer can show several
 *  sessions at once (tiles) but there is one speaker, so playback is a single
 *  slot — this is how each surface tells whose slot it is. */
export function voicePlaybackBelongsTo(
  playback: VoicePlaybackState,
  sessionId: string | null | undefined
): boolean {
  return playback.sessionId === voicePlaybackSessionKey(sessionId)
}

export function setVoicePlaybackState(next: VoicePlaybackState) {
  $voicePlayback.set(next)
}
