import { atom } from 'nanostores'

import { persistBoolean, readKey, storedBoolean } from '@/lib/storage'

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

// Backend default for `voice.stop_phrases` (DEFAULT_VOICE_STOP_PHRASES in
// tools/voice_mode_transcript.py), used until `/api/config/defaults` answers.
const BACKEND_DEFAULT_STOP_PHRASES: readonly string[] = ['stop']

// First configured `voice.stop_phrases` entry — drives the "Say "stop" to end
// the voice chat" notice shown when a voice conversation starts. `null` means
// the user disabled stop phrases (`stop_phrases: []`), so no notice is shown.
// Defaults to "stop" (the backend default) before config loads.
export const $voiceStopPhrase = atom<string | null>(BACKEND_DEFAULT_STOP_PHRASES[0])

/** How the desktop matcher should treat `voice.stop_phrases` (#117801). */
export type VoiceStopPhraseConfig =
  { mode: 'default' } | { mode: 'custom'; phrases: readonly string[] } | { mode: 'disabled' }

// Full matcher config — kept in sync with `$voiceStopPhrase` so the spoken
// stop recognizer honours the same list the notice advertises.
export const $voiceStopPhraseConfig = atom<VoiceStopPhraseConfig>({ mode: 'default' })

type ConfigPayload = Record<string, unknown> | { voice?: object | null } | null | undefined

function voiceValue(config: ConfigPayload, key: string): unknown {
  const voice = (config as { voice?: unknown } | null | undefined)?.voice

  return voice && typeof voice === 'object' ? (voice as Record<string, unknown>)[key] : undefined
}

/** Parse `voice.stop_phrases` like `_load_voice_stop_phrases`: a bare string
 *  counts as one phrase, non-text entries are skipped, and anything that is not
 *  a list or string (absent, null, a mapping) means "backend default" (`null`). */
function stopPhraseList(raw: unknown): string[] | null {
  const list = typeof raw === 'string' ? [raw] : Array.isArray(raw) ? raw : null

  return (
    list
      ?.filter(entry => typeof entry === 'string' || typeof entry === 'number')
      .map(entry => String(entry).trim())
      .filter(entry => entry.length > 0) ?? null
  )
}

function samePhrases(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((phrase, i) => phrase.toLowerCase() === b[i].toLowerCase())
}

/**
 * Seed the stop-phrase atoms from a loaded config payload (mount / refresh).
 *
 * `/api/config` merges DEFAULT_CONFIG, so an untouched install reports the
 * backend default (`["stop"]`) rather than omitting the key. Pass the
 * `/api/config/defaults` payload so that case keeps the built-in English
 * matcher list; only a list the user actually changed replaces it.
 */
export function applyVoiceStopPhraseFromConfig(config: ConfigPayload, defaults?: ConfigPayload) {
  const phrases = stopPhraseList(voiceValue(config, 'stop_phrases'))

  if (phrases === null) {
    // Key absent or malformed — backend default + built-in English matcher list.
    $voiceStopPhrase.set(BACKEND_DEFAULT_STOP_PHRASES[0])
    $voiceStopPhraseConfig.set({ mode: 'default' })

    return
  }

  const defaultPhrases = stopPhraseList(voiceValue(defaults, 'stop_phrases')) ?? BACKEND_DEFAULT_STOP_PHRASES

  $voiceStopPhrase.set(phrases[0] ?? null)
  $voiceStopPhraseConfig.set(
    phrases.length === 0
      ? { mode: 'disabled' }
      : samePhrases(phrases, defaultPhrases)
        ? { mode: 'default' }
        : { mode: 'custom', phrases }
  )
}

// `voice.barge_in_threshold_multiplier` — barge-in sensitivity shared with the
// CLI/TUI full-duplex listener. `null` = unset/invalid, so the monitor keeps its
// built-in trigger levels (same as the backend's `float(v or 0)` → default).
export const $bargeInThresholdMultiplier = atom<number | null>(null)

/** Seed the barge-in sensitivity from a loaded config payload. */
export function applyBargeInThresholdFromConfig(config: ConfigPayload) {
  const value = Number(voiceValue(config, 'barge_in_threshold_multiplier'))

  $bargeInThresholdMultiplier.set(Number.isFinite(value) && value > 0 ? value : null)
}

// `voice.silence_duration` (seconds) — how long the user must stay quiet
// before the conversation loop treats the utterance as finished. Documented
// default 3.0 (hermes_cli/config_defaults.py), honoured by the CLI/TUI/gateway
// capture paths (cli_voice_mixin.py, tui_gateway/methods_voice.py) but
// previously hardcoded to 1.25 s in the desktop renderer's mic loop, so a
// mid-thought pause cut the turn off and `hermes config set` had no effect.
// Stored in ms because that is what the loop's timers consume.
//
// `/api/config` merges DEFAULT_CONFIG, so an untouched install reports the
// backend default (3.0) rather than omitting the key; only a value the user
// actually changed (≠ the /api/config/defaults payload) overrides the
// desktop's tuned default — reading it unconditionally would triple the hold
// for everyone who never touched the key.
const DESKTOP_SILENCE_MS_DEFAULT = 1_250
const BACKEND_SILENCE_SECONDS_DEFAULT = 3.0

export const $voiceSilenceMs = atom<number>(DESKTOP_SILENCE_MS_DEFAULT)

function silenceSeconds(raw: unknown): number | null {
  // `true` must not coerce to 1 s — YAML hands back a bare boolean for a typo'd value.
  const value = typeof raw === 'number' ? raw : typeof raw === 'boolean' ? NaN : Number(raw)

  return Number.isFinite(value) && value > 0 ? value : null
}

/** Seed the silence window from a loaded config payload (mount / refresh). */
export function applyVoiceSilenceMsFromConfig(config: ConfigPayload, defaults?: ConfigPayload) {
  const seconds = silenceSeconds(voiceValue(config, 'silence_duration'))
  const defaultSeconds = silenceSeconds(voiceValue(defaults, 'silence_duration')) ?? BACKEND_SILENCE_SECONDS_DEFAULT

  $voiceSilenceMs.set(seconds !== null && seconds !== defaultSeconds ? seconds * 1_000 : DESKTOP_SILENCE_MS_DEFAULT)
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
