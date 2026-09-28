/**
 * What the orb should show while a desktop voice conversation runs.
 *
 * Live voice (Gemini Live, OpenAI Realtime) drives its whole state machine in
 * the renderer, so the gateway's `voice.status` events — the source the classic
 * CLI loop feeds — never arrive for it. Without this seam the orb would stay on
 * whatever the last gateway event left behind (idle) while Gemini is actually
 * speaking. The classic desktop loop reports through the same seam, so both
 * engines move the orb the same way.
 */

import type { ConversationStatus } from '@/app/chat/composer/hooks/use-voice-conversation'

import type { JarvisVoiceState } from './types'

export function desktopOrbVoice(status: ConversationStatus, active: boolean): JarvisVoiceState {
  if (!active) {
    return 'idle'
  }

  if (status === 'speaking') {
    return 'speaking'
  }

  // `thinking` means the agent is working on an `ask_jarvis` request, not the
  // voice: the backend's task phase already drives the orb then (planning →
  // thinking, running → working), so the voice must not claim the microphone.
  if (status === 'thinking') {
    return 'idle'
  }

  return 'listening'
}
