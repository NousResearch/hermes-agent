import { atom } from 'nanostores'

import { initialJarvisUiState, type JarvisEvent, reduceJarvisEvent } from './projector'
import type { JarvisVoiceState } from './types'

export const $jarvisUi = atom(initialJarvisUiState())

export const publishJarvisEvent = (event: JarvisEvent): void => {
  $jarvisUi.set(reduceJarvisEvent($jarvisUi.get(), event))
}

/**
 * The orb's voice state, reported by whoever owns the conversation.
 *
 * The gateway's `voice.status` events only exist for the classic CLI loop; the
 * desktop's own voice conversation (Live voice above all) lives in the renderer
 * and tells the orb directly. Unlike `publishJarvisEvent` this is not a session
 * event — a voice status belongs to the window, not to one chat — so it must
 * never go through the projector's session/ordering filters.
 */
export const publishJarvisVoiceState = (voice: JarvisVoiceState): void => {
  const state = $jarvisUi.get()

  if (state.voice !== voice) {
    $jarvisUi.set({ ...state, voice })
  }
}

export const resetJarvisSession = (sessionId: string | null): void => {
  $jarvisUi.set({ ...initialJarvisUiState(), sessionId })
}
