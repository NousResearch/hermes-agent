// Session isolation for one renderer's single playback slot.
//
// The desktop can show several sessions at once (tiles) in ONE renderer, and
// the speaker is not a per-session device: whichever session spoke last owns
// the slot. Without an owner recorded on the state, every composer pill and
// every message Read-aloud button reacted to a sibling session's clip — one
// click "read aloud" in one chat lit every open chat and disabled every other
// message's button.

import { describe, expect, it } from 'vitest'

import {
  $voicePlayback,
  setVoicePlaybackState,
  voicePlaybackBelongsTo,
  voicePlaybackSessionKey,
  type VoicePlaybackState
} from './voice-playback'

const idle = (sessionId: string | null = null): VoicePlaybackState => ({
  audioElement: null,
  messageId: null,
  sequence: 0,
  sessionId,
  source: null,
  status: 'idle'
})

const speaking = (sessionId: string | null, messageId = 'm1'): VoicePlaybackState => ({
  ...idle(sessionId),
  messageId,
  sequence: 1,
  source: 'read-aloud',
  status: 'speaking'
})

describe('voice playback session ownership', () => {
  it('normalizes a fresh chat (empty id) and a missing id to the same key', () => {
    expect(voicePlaybackSessionKey('')).toBeNull()
    expect(voicePlaybackSessionKey(undefined)).toBeNull()
    expect(voicePlaybackSessionKey(null)).toBeNull()
    expect(voicePlaybackSessionKey('session-a')).toBe('session-a')
  })

  it('attributes a clip to the session that started it, and to no other', () => {
    const state = speaking('session-a')

    expect(voicePlaybackBelongsTo(state, 'session-a')).toBe(true)
    // The reported bug: a sibling chat must NOT own the clip.
    expect(voicePlaybackBelongsTo(state, 'session-b')).toBe(false)
    expect(voicePlaybackBelongsTo(state, null)).toBe(false)
  })

  it('matches a fresh chat against a clip that carried no session id', () => {
    expect(voicePlaybackBelongsTo(speaking(null), '')).toBe(true)
    expect(voicePlaybackBelongsTo(speaking(null), undefined)).toBe(true)
    expect(voicePlaybackBelongsTo(speaking(null), 'session-a')).toBe(false)
  })

  it('keeps the stopped clip owner on the idle state so a session can tell its own clip was cut', () => {
    setVoicePlaybackState(speaking('session-a'))
    const stopped = idle($voicePlayback.get().sessionId)

    setVoicePlaybackState(stopped)

    expect(stopped.status).toBe('idle')
    // Still the owner: callers that watch `sequence` to detect "MY clip was
    // cut" must be able to attribute the stop to session-a, not to session-b.
    expect(voicePlaybackBelongsTo(stopped, 'session-a')).toBe(true)
    expect(voicePlaybackBelongsTo(stopped, 'session-b')).toBe(false)
  })
})
