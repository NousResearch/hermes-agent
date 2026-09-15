import { atom } from 'nanostores'

import { fetchVoiceLiveStatus, type VoiceChatMode, type VoiceLiveStatus } from '@/lib/voice-live'
import { fetchVoiceLiveGrokStatus } from '@/lib/voice-live-grok'
import { activeGateway } from '@/store/gateway'

/**
 * `voice.voice_chat_mode` as the backend resolves it, plus whether GPT-Live can
 * actually start (an OpenAI key resolves on the gateway host). The composer
 * mounts the chained, GPT-Live or Grok-Live conversation engine from this;
 * refreshed with the config snapshot so a Settings change applies to the next
 * conversation.
 *
 * Grok-Live gets its OWN status atom (`$voiceLiveGrokStatus`) rather than a
 * widened `VoiceLiveStatus` on this one — SPEC §11's deliberate choice to add
 * a parallel status route instead of retrofitting the existing gpt-live-only
 * endpoint mid-migration. Both atoms report the SAME `voice.voice_chat_mode`
 * value (the shared backend resolver), so `selectedVoiceChatMode` reads
 * whichever one answered.
 */
export const $voiceLiveStatus = atom<null | VoiceLiveStatus>(null)
export const $voiceLiveGrokStatus = atom<null | VoiceLiveStatus>(null)

let inflight: null | Promise<null | VoiceLiveStatus> = null
let grokInflight: null | Promise<null | VoiceLiveStatus> = null

export async function refreshVoiceLiveStatus(): Promise<null | VoiceLiveStatus> {
  if (inflight) {
    return inflight
  }

  inflight = fetchVoiceLiveStatus()
    .then(status => {
      $voiceLiveStatus.set(status)

      return status
    })
    .finally(() => {
      inflight = null
    })

  return inflight
}

export async function refreshVoiceLiveGrokStatus(): Promise<null | VoiceLiveStatus> {
  if (grokInflight) {
    return grokInflight
  }

  grokInflight = fetchVoiceLiveGrokStatus()
    .then(status => {
      $voiceLiveGrokStatus.set(status)

      return status
    })
    .finally(() => {
      grokInflight = null
    })

  return grokInflight
}

/** Refresh both status sources from the same call sites (config-change handler,
 *  conversation-mount effect) so they never drift out of sync (SPEC §13 risk 7). */
export async function refreshAllVoiceLiveStatuses(): Promise<void> {
  await Promise.all([refreshVoiceLiveStatus(), refreshVoiceLiveGrokStatus()])
}

/** Selected mode. `chained` until the backend answers, or when the backend predates the mode. */
export function selectedVoiceChatMode(status: null | VoiceLiveStatus = $voiceLiveStatus.get()): VoiceChatMode {
  return status?.mode === 'gpt-live' || status?.mode === 'grok-live' ? status.mode : 'chained'
}

/**
 * Persist `voice.voice_chat_mode` on the live gateway (whichever profile/host
 * the app is talking to) and re-read the resolved status, so the menu shows
 * what the backend will actually mount next. Takes effect on the NEXT
 * conversation; an active one keeps its engine.
 */
export async function setVoiceChatMode(mode: VoiceChatMode): Promise<null | VoiceLiveStatus> {
  const gateway = activeGateway()

  if (!gateway) {
    throw new Error('gateway not connected')
  }

  await gateway.request('config.set', { key: 'voice.voice_chat_mode', value: mode })
  await refreshAllVoiceLiveStatuses()

  return $voiceLiveStatus.get()
}
