import { atom, computed } from 'nanostores'

import { persistBoolean, readKey, storedBoolean } from '@/lib/storage'
import { DEFAULT_VOICE_STOP_PHRASES } from '@/lib/voice-stop-word'

// Desktop read-aloud is local; voice.auto_tts belongs to the messaging gateway.
const AUTO_SPEAK_KEY = 'hermes.desktop.autoSpeakReplies'
export const $autoSpeakReplies = atom<boolean>(storedBoolean(AUTO_SPEAK_KEY, false))
// Best-effort persistence must not give config refresh authority again.
let autoSpeakChosen = readKey(AUTO_SPEAK_KEY) !== null

/** Migrate the legacy value once without editing the backend configuration. */
export function applyAutoSpeakFromConfig(config: { voice?: { auto_tts?: unknown } | null } | null | undefined) {
  if (config != null && !autoSpeakChosen) {
    void setAutoSpeakReplies(Boolean(config.voice?.auto_tts))
  }
}

// First configured `voice.stop_phrases` entry — drives the "Say "stop" to end
// the voice chat" notice shown when a voice conversation starts. `null` means
// the user disabled stop phrases (`stop_phrases: []`), so no notice is shown.
// Defaults to "stop" (the backend default) before config loads.
export const $voiceStopPhrases = atom<readonly string[]>(DEFAULT_VOICE_STOP_PHRASES)
export const $voiceStopPhrase = computed($voiceStopPhrases, phrases => phrases[0] ?? null)

/** Seed the stop-phrase atom from a loaded config payload (mount / refresh). */
export function applyVoiceStopPhraseFromConfig(
  config: { voice?: { stop_phrases?: unknown } | null } | null | undefined
) {
  const raw = config?.voice?.stop_phrases

  if (!Array.isArray(raw) && typeof raw !== 'string') {
    // Missing or malformed containers use the backend default.
    $voiceStopPhrases.set(DEFAULT_VOICE_STOP_PHRASES)

    return
  }

  const list = Array.isArray(raw) ? raw : [raw]

  const phrases = list
    .filter(entry => typeof entry === 'string' || typeof entry === 'number')
    .map(entry => String(entry).trim())
    .filter(Boolean)

  $voiceStopPhrases.set(phrases)
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

/** Persist even an unchanged value, so migrating false is also one-time. */
export async function setAutoSpeakReplies(enabled: boolean): Promise<void> {
  autoSpeakChosen = true
  persistBoolean(AUTO_SPEAK_KEY, enabled)
  $autoSpeakReplies.set(enabled)
}
