import { atom } from 'nanostores'

import { fetchVoiceLiveStatus, type VoiceLiveStatus } from '@/lib/voice-live'
import { activeGateway } from '@/store/gateway'

/**
 * `voice.voice_chat_mode` as the backend resolves it, plus whether GPT-Live can
 * actually start (an OpenAI key resolves on the gateway host). The composer
 * mounts the chained or the live conversation engine from this; refreshed with
 * the config snapshot so a Settings change applies to the next conversation.
 */
export const $voiceLiveStatus = atom<null | VoiceLiveStatus>(null)

let inflight: null | Promise<null | VoiceLiveStatus> = null

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

export type DesktopVoiceChatMode = 'chained' | 'gpt-live' | 'gemini-live'
export const HERMES_DESKTOP_VOICE_MODE_STORAGE = 'hermes_desktop_voice_mode'

/** Selected mode. Falls back to local desktop preference or backend status. */
export function selectedVoiceChatMode(status: null | VoiceLiveStatus = $voiceLiveStatus.get()): DesktopVoiceChatMode {
  try {
    const local = localStorage.getItem(HERMES_DESKTOP_VOICE_MODE_STORAGE)
    if (local === 'gemini-live' || local === 'gpt-live' || local === 'chained') {
      return local
    }
  } catch {}

  return status?.mode === 'gpt-live' ? 'gpt-live' : 'chained'
}

/**
 * Persist voice chat mode. Stored in client localStorage so desktop-only
 * modes like gemini-live work without error even if the remote VPS gateway
 * has an older schema or rejects the config key.
 */
export async function setVoiceChatMode(mode: DesktopVoiceChatMode): Promise<null | VoiceLiveStatus> {
  try {
    localStorage.setItem(HERMES_DESKTOP_VOICE_MODE_STORAGE, mode)
  } catch {}

  // Also sync to remote gateway if it's one of the gateway-supported modes
  if (mode === 'chained' || mode === 'gpt-live') {
    try {
      const gateway = activeGateway()
      if (gateway) {
        await gateway.request('config.set', { key: 'voice.voice_chat_mode', value: mode })
      }
    } catch {}
  }

  const current = $voiceLiveStatus.get()
  if (current) {
    $voiceLiveStatus.set({ ...current, mode: mode as any })
  }

  return refreshVoiceLiveStatus()
}

export const $geminiLiveDialogOpen = atom<boolean>(false)

export function openGeminiLiveDialog(): void {
  $geminiLiveDialogOpen.set(true)
}

export function closeGeminiLiveDialog(): void {
  $geminiLiveDialogOpen.set(false)
}
