import { atom } from 'nanostores'

import { getHermesConfigRecord, saveHermesConfig } from '@/hermes'

// "Read replies aloud" — mirrors the canonical `voice.auto_tts` config key (also
// in Settings → Voice, honored by the messaging gateway) so the composer toggle
// and the Settings switch are one source of truth, not two that can disagree.
export const $autoSpeakReplies = atom<boolean>(false)
export type VoiceInputMode = 'legacy' | 'realtime'
export const $voiceInputMode = atom<VoiceInputMode>('legacy')

/** Seed the atom from a loaded config payload (mount / refresh). */
export function applyAutoSpeakFromConfig(
  config: { voice?: { auto_tts?: unknown; input_mode?: unknown } | null } | null | undefined
) {
  $autoSpeakReplies.set(Boolean(config?.voice?.auto_tts))
  $voiceInputMode.set(config?.voice?.input_mode === 'realtime' ? 'realtime' : 'legacy')
}

/**
 * Flip the preference and persist it. Optimistic — the atom updates instantly and
 * reverts if the config write fails. Read-modify-writes the whole record (the
 * same path the Settings page uses; there's no partial-update endpoint).
 */
export async function setAutoSpeakReplies(enabled: boolean): Promise<void> {
  const previous = $autoSpeakReplies.get()

  if (previous === enabled) {
    return
  }

  $autoSpeakReplies.set(enabled)

  try {
    const record = await getHermesConfigRecord()
    const voice = record.voice && typeof record.voice === 'object' ? (record.voice as Record<string, unknown>) : {}

    await saveHermesConfig({ ...record, voice: { ...voice, auto_tts: enabled } })
  } catch (error) {
    $autoSpeakReplies.set(previous)
    throw error
  }
}

/** Select and persist the Desktop input transport. Realtime is explicit opt-in. */
export async function setVoiceInputMode(mode: VoiceInputMode): Promise<void> {
  const previous = $voiceInputMode.get()

  if (previous === mode) {
    return
  }

  const record = await getHermesConfigRecord()
  const voice = record.voice && typeof record.voice === 'object' ? (record.voice as Record<string, unknown>) : {}

  const existingRealtime =
    voice.realtime && typeof voice.realtime === 'object' ? (voice.realtime as Record<string, unknown>) : {}

  await saveHermesConfig({
    ...record,
    voice: {
      ...voice,
      input_mode: mode,
      realtime: { ...existingRealtime, enabled: mode === 'realtime' }
    }
  })
  // Publish only after the backend config is durable; otherwise a fast click
  // on Start can race the feature gate and see a misleading disabled error.
  $voiceInputMode.set(mode)
}
