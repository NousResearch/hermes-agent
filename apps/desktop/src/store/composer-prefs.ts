/**
 * Composer send behaviour (Settings → Keyboard Shortcuts → Send behavior): what a
 * bare Enter does, which gestures also commit a draft, and how long each window is.
 *
 * The gateway config owns the values, as `desktop.composer.*` in config.yaml, so
 * they can be hand-edited with more precision than the controls offer. This module
 * is the renderer's mirror of that record, and it re-exports the shared contract so
 * callers have one import for both the atoms and the constants.
 */

import {
  activeSendGestures,
  clampDoubleEnterMs,
  clampHoldMs,
  clampIdleSendMs,
  clampSendGraceMs,
  clampTypingIdleMs,
  COMPOSER_SEND_GESTURES,
  composerPrefsFromConfig,
  type ComposerSendGesture,
  type ComposerSendPrefs,
  DOUBLE_ENTER_DEFAULT_MS,
  DOUBLE_ENTER_MAX_MS,
  DOUBLE_ENTER_MIN_MS,
  HOLD_DEFAULT_MS,
  HOLD_MAX_MS,
  HOLD_MIN_MS,
  IDLE_SEND_DEFAULT_MS,
  IDLE_SEND_MAX_MS,
  IDLE_SEND_MIN_MS,
  SEND_GRACE_DEFAULT_MS,
  SEND_GRACE_DEFAULT_REASONS,
  SEND_GRACE_MAX_MS,
  SEND_GRACE_MIN_MS,
  SEND_GRACE_REASONS,
  type SendGraceReason,
  TYPING_IDLE_DEFAULT_MS,
  TYPING_IDLE_MAX_MS,
  TYPING_IDLE_MIN_MS
} from '@hermes/shared'
import { atom } from 'nanostores'

/** The full contract: the gate, every gesture, and each window. One atom, so
 *  the gate the keydown handler consults cannot disagree with the prefs the
 *  settings page renders. */
export const $composerSendPrefs = atom<ComposerSendPrefs>(composerPrefsFromConfig({}))

/** Mirror `desktop.composer` out of a config record. */
export function applyComposerPrefsFromConfig(config: {
  desktop?: { composer?: Record<string, unknown> }
}): void {
  $composerSendPrefs.set(composerPrefsFromConfig(config.desktop?.composer))
}

export type { ComposerSendGesture, ComposerSendPrefs, SendGraceReason }

export {
  activeSendGestures,
  clampDoubleEnterMs,
  clampHoldMs,
  clampIdleSendMs,
  clampSendGraceMs,
  clampTypingIdleMs,
  COMPOSER_SEND_GESTURES,
  composerPrefsFromConfig,
  DOUBLE_ENTER_DEFAULT_MS,
  DOUBLE_ENTER_MAX_MS,
  DOUBLE_ENTER_MIN_MS,
  HOLD_DEFAULT_MS,
  HOLD_MAX_MS,
  HOLD_MIN_MS,
  IDLE_SEND_DEFAULT_MS,
  IDLE_SEND_MAX_MS,
  IDLE_SEND_MIN_MS,
  SEND_GRACE_DEFAULT_MS,
  SEND_GRACE_DEFAULT_REASONS,
  SEND_GRACE_MAX_MS,
  SEND_GRACE_MIN_MS,
  SEND_GRACE_REASONS,
  TYPING_IDLE_DEFAULT_MS,
  TYPING_IDLE_MAX_MS,
  TYPING_IDLE_MIN_MS
}
