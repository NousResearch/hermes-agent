import { atom } from 'nanostores'

import { persistStore } from '@/lib/persist-store'
import { persistString, storedString } from '@/lib/storage'

const STORAGE_KEY = 'hermes.desktop.completionSoundVariantId'

export const DEFAULT_COMPLETION_SOUND_VARIANT_ID = 1

// "None": variant 0 disables the turn-end cue but stays a selectable settings value.
export const NO_COMPLETION_SOUND_VARIANT_ID = 0

// Range mirrors COMPLETION_SOUND_VARIANTS in lib/completion-sound.ts (0 = no
// sound). Validating by range (not membership) keeps this store free of a
// dependency on the lib, which imports the atom back — a membership check would
// close that cycle.
const VARIANT_COUNT = 14

export function resolveCompletionSoundVariantId(variantId: number): number {
  return Number.isInteger(variantId) && variantId >= 0 && variantId <= VARIANT_COUNT
    ? variantId
    : DEFAULT_COMPLETION_SOUND_VARIANT_ID
}

function load(): number {
  const stored = storedString(STORAGE_KEY)

  return stored ? resolveCompletionSoundVariantId(Number.parseInt(stored, 10)) : DEFAULT_COMPLETION_SOUND_VARIANT_ID
}

export const $completionSoundVariantId = atom(load())

persistStore(
  $completionSoundVariantId,
  (id, options) => persistString(STORAGE_KEY, String(id), options),
  DEFAULT_COMPLETION_SOUND_VARIANT_ID
)

export function setCompletionSoundVariantId(variantId: number) {
  $completionSoundVariantId.set(resolveCompletionSoundVariantId(variantId))
}

// ── Volume ───────────────────────────────────────────────────────────────────

const VOLUME_STORAGE_KEY = 'hermes.desktop.completionSoundVolume'

/** The master gain this setting replaced. Existing users hear no change on
 *  update because the persisted default is exactly the old hardcoded value. */
export const DEFAULT_COMPLETION_SOUND_VOLUME = 0.48

/** 0 is a mute, 3 is a loud boost. A parsed number outside the range is clamped
 *  — 3.4 is a loud user, not a corrupt one — while a stored value that is not a
 *  plain decimal at all falls back to the default, so a mangled key can never
 *  turn into max gain or a mute. */
const VOLUME_MIN = 0
const VOLUME_MAX = 3

// A plain decimal only: rejects '3.0.1', '1.5abc', '0x10' and '1e9', all of
// which `Number.parseFloat`/`Number` would otherwise launder into a valid gain.
const DECIMAL = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/

function clampVolume(volume: number): number {
  return Math.min(Math.max(volume, VOLUME_MIN), VOLUME_MAX)
}

export function resolveCompletionSoundVolume(raw: number): number {
  return Number.isFinite(raw) ? clampVolume(raw) : DEFAULT_COMPLETION_SOUND_VOLUME
}

function loadVolume(): number {
  const stored = storedString(VOLUME_STORAGE_KEY)

  if (stored === null) {
    return DEFAULT_COMPLETION_SOUND_VOLUME
  }

  const trimmed = stored.trim()

  return DECIMAL.test(trimmed) ? resolveCompletionSoundVolume(Number(trimmed)) : DEFAULT_COMPLETION_SOUND_VOLUME
}

export const $completionSoundVolume = atom(loadVolume())

persistStore(
  $completionSoundVolume,
  (volume, options) => persistString(VOLUME_STORAGE_KEY, String(volume), options),
  DEFAULT_COMPLETION_SOUND_VOLUME
)

// Keep the cue's volume in step across open windows, like the mute toggle: the
// window that owns a cue is not necessarily the one showing settings.
if (typeof window !== 'undefined') {
  window.addEventListener('storage', event => {
    if (event.key === VOLUME_STORAGE_KEY) {
      $completionSoundVolume.set(loadVolume())
    }
  })
}

export function setCompletionSoundVolume(volume: number) {
  $completionSoundVolume.set(resolveCompletionSoundVolume(volume))
}
