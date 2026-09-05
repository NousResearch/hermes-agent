import { atom } from 'nanostores'

// "Read replies aloud" — a desktop-local preference. It deliberately does NOT
// read or write `voice.auto_tts`: that key also drives the messaging gateway's
// auto-TTS, so sharing it made the gateway voice bubble and desktop auto-speak
// fire together (double-read, #99076). The first run with no stored preference
// migrates from the old shared value once, so existing setups keep their state.
const AUTO_SPEAK_KEY = 'hermes.desktop.autoSpeakReplies'

function readStoredAutoSpeak(): boolean | null {
  try {
    const raw = window.localStorage.getItem(AUTO_SPEAK_KEY)

    return raw === null ? null : raw === 'true'
  } catch {
    // Storage unavailable (locked-down renderer, quota edge) — fall back to
    // treating the preference as unset rather than crashing module load.
    return null
  }
}

export const $autoSpeakReplies = atom<boolean>(readStoredAutoSpeak() ?? false)

/**
 * Seed the atom from a loaded config payload (mount / refresh). Only migrates
 * from the legacy shared `voice.auto_tts` value while this desktop has never
 * stored its own preference — after that, gateway-side changes to
 * `voice.auto_tts` must not flip the local toggle.
 */
export function applyAutoSpeakFromConfig(config: { voice?: { auto_tts?: unknown } | null } | null | undefined) {
  if (readStoredAutoSpeak() !== null) {
    return
  }

  $autoSpeakReplies.set(Boolean(config?.voice?.auto_tts))
}

// First configured `voice.stop_phrases` entry — drives the "Say "stop" to end
// the voice chat" notice shown when a voice conversation starts. `null` means
// the user disabled stop phrases (`stop_phrases: []`), so no notice is shown.
// Defaults to "stop" (the backend default) before config loads.
export const $voiceStopPhrase = atom<string | null>('stop')

/** Seed the stop-phrase atom from a loaded config payload (mount / refresh). */
export function applyVoiceStopPhraseFromConfig(
  config: { voice?: { stop_phrases?: unknown } | null } | null | undefined
) {
  const raw = config?.voice?.stop_phrases

  if (raw === undefined) {
    // Key absent — backend default applies.
    $voiceStopPhrase.set('stop')

    return
  }

  const list = Array.isArray(raw) ? raw : typeof raw === 'string' ? [raw] : []
  const first = list.map(entry => String(entry).trim()).find(entry => entry.length > 0)

  $voiceStopPhrase.set(first ?? null)
}

// `voice.thinking_sound` — ambient bubble blips while the agent works during a
// voice conversation (default on, matching the backend default).
export const $thinkingSoundEnabled = atom<boolean>(true)

/** Seed the thinking-sound gate from a loaded config payload. */
export function applyThinkingSoundFromConfig(
  config: { voice?: { thinking_sound?: unknown } | null } | null | undefined
) {
  $thinkingSoundEnabled.set(config?.voice?.thinking_sound !== false)
}

/**
 * Flip the preference and persist it locally. Optimistic — the atom updates
 * instantly and reverts if the write fails. Never touches the shared config:
 * `voice.auto_tts` stays owned by Settings → Voice and the gateway.
 */
export async function setAutoSpeakReplies(enabled: boolean): Promise<void> {
  const previous = $autoSpeakReplies.get()

  if (previous === enabled) {
    return
  }

  $autoSpeakReplies.set(enabled)

  try {
    window.localStorage.setItem(AUTO_SPEAK_KEY, String(enabled))
  } catch (error) {
    $autoSpeakReplies.set(previous)
    throw error
  }
}
