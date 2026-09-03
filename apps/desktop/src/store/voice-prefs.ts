import { atom } from 'nanostores'

import { getHermesConfigRecord, saveHermesConfig } from '@/hermes'

// "Read replies aloud" — mirrors the canonical `voice.auto_tts` config key (also
// in Settings → Voice, honored by the messaging gateway) so the composer toggle
// and the Settings switch are one source of truth, not two that can disagree.
export const $autoSpeakReplies = atom<boolean>(false)

/** Seed the atom from a loaded config payload (mount / refresh). */
export function applyAutoSpeakFromConfig(config: { voice?: { auto_tts?: unknown } | null } | null | undefined) {
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

// `voice.silence_duration` (seconds) — how long the user must stay quiet before
// the conversation loop treats the utterance as finished. This was hardcoded to
// 1250 ms in the loop, so the config key existed but only ever drove the CLI
// path (tools/voice_mode.py); the desktop never consulted it. Every millisecond
// here is dead air the user sits through on EVERY turn, so it is the cheapest
// latency in the whole voice pipeline to tune.
//
// Clamped: below ~300 ms normal mid-sentence pauses cut the user off, and above
// ~3 s the conversation stops feeling responsive at all.
const SILENCE_MS_DEFAULT = 700
const SILENCE_MS_MIN = 300
const SILENCE_MS_MAX = 3_000

export const $voiceSilenceMs = atom<number>(SILENCE_MS_DEFAULT)

/** Seed the silence window from a loaded config payload (mount / refresh). */
export function applyVoiceSilenceFromConfig(
  config: { voice?: { silence_duration?: unknown } | null } | null | undefined
) {
  const raw = config?.voice?.silence_duration
  const seconds = typeof raw === 'number' ? raw : Number.parseFloat(String(raw ?? ''))

  if (!Number.isFinite(seconds) || seconds <= 0) {
    $voiceSilenceMs.set(SILENCE_MS_DEFAULT)

    return
  }

  $voiceSilenceMs.set(Math.min(SILENCE_MS_MAX, Math.max(SILENCE_MS_MIN, Math.round(seconds * 1_000))))
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
