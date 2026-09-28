import { atom } from 'nanostores'

import { fetchVoiceLiveStatus, type VoiceLiveStatus } from '@/lib/voice-live'
import { activeGateway } from '@/store/gateway'

import { profileScopeKey } from '@/hermes'
import { resolveGeminiLiveApiKey } from '@/lib/gemini-live'

/**
 * `voice.voice_chat_mode` as the backend resolves it, plus whether GPT-Live can
 * actually start (an OpenAI key resolves on the gateway host). The composer
 * mounts the chained or the live conversation engine from this; refreshed with
 * the config snapshot so a Settings change applies to the next conversation.
 */
export const $voiceLiveStatus = atom<null | VoiceLiveStatus>(null)
export const $geminiLiveKeyConfigured = atom<boolean | null>(null)

export async function refreshGeminiLiveKeyStatus(): Promise<boolean> {
  try {
    const key = await resolveGeminiLiveApiKey()
    const configured = Boolean(key)
    $geminiLiveKeyConfigured.set(configured)
    return configured
  } catch {
    $geminiLiveKeyConfigured.set(false)
    return false
  }
}

let inflight: null | Promise<null | VoiceLiveStatus> = null

export async function refreshVoiceLiveStatus(): Promise<null | VoiceLiveStatus> {
  if (inflight) {
    return inflight
  }

  void refreshGeminiLiveKeyStatus().catch(() => undefined)

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
export const HERMES_DESKTOP_GEMINI_LIVE_STORAGE_PREFIX = 'hermes_desktop_gemini_live:'

function getScopedGeminiLiveStorageKey(): string {
  try {
    const scope = profileScopeKey()
    return `${HERMES_DESKTOP_GEMINI_LIVE_STORAGE_PREFIX}${scope}`
  } catch {
    return `${HERMES_DESKTOP_GEMINI_LIVE_STORAGE_PREFIX}default`
  }
}

/**
 * Returns the active voice chat mode.
 * Backend-owned modes ('chained', 'gpt-live') remain authoritative from status.mode.
 * The desktop-only 'gemini-live' engine is scoped to the active connection/profile.
 */
export function selectedVoiceChatMode(status: null | VoiceLiveStatus = $voiceLiveStatus.get()): DesktopVoiceChatMode {
  // Purge any legacy unscoped voice mode key
  try {
    localStorage.removeItem('hermes_desktop_voice_mode')
  } catch {}

  try {
    const key = getScopedGeminiLiveStorageKey()
    if (localStorage.getItem(key) === '1') {
      return 'gemini-live'
    }
  } catch {}

  return status?.mode === 'gpt-live' ? 'gpt-live' : 'chained'
}

/**
 * Persist voice chat mode.
 * Selecting 'gemini-live' sets the desktop-only preference for the active profile scope.
 * Selecting 'chained' or 'gpt-live' clears the desktop override and synchronizes with the gateway.
 */
export async function setVoiceChatMode(mode: DesktopVoiceChatMode): Promise<null | VoiceLiveStatus> {
  const scopedKey = getScopedGeminiLiveStorageKey()

  // Sync backend-supported modes to the active gateway, propagating any connection/config errors
  if (mode === 'chained' || mode === 'gpt-live') {
    const gateway = activeGateway()
    if (!gateway) {
      throw new Error('gateway not connected')
    }
    await gateway.request('config.set', { key: 'voice.voice_chat_mode', value: mode })

    try {
      localStorage.removeItem(scopedKey)
    } catch {}
  } else if (mode === 'gemini-live') {
    try {
      localStorage.setItem(scopedKey, '1')
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
