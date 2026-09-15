/**
 * Composer send behaviour (Settings → Keyboards): which keypress commits a
 * draft, and how wide the `double-enter` window is.
 *
 * Main owns the persisted value — a small JSON file under userData, so the
 * window can be hand-edited with more precision than the slider offers (see
 * electron/composer-send-ipc.ts). This atom mirrors it for the composer
 * keydown handler and the settings UI. Defaults and bounds are shared with main
 * via apps/shared so the two ends cannot drift.
 */

import {
  clampDoubleEnterMs,
  clampHoldMs,
  clampSendGraceMs,
  clampTypingIdleMs,
  COMPOSER_SEND_DEFAULT_MODE,
  type ComposerSendMode,
  type ComposerSendPrefs,
  DOUBLE_ENTER_DEFAULT_MS,
  DOUBLE_ENTER_MAX_MS,
  DOUBLE_ENTER_MIN_MS,
  HOLD_MAX_MS,
  HOLD_MIN_MS,
  normalizeComposerSendPrefs,
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
import { atom, computed } from 'nanostores'

import { notifyError } from '@/store/notifications'

export {
  clampDoubleEnterMs,
  clampHoldMs,
  clampSendGraceMs,
  clampTypingIdleMs,
  COMPOSER_SEND_DEFAULT_MODE,
  type ComposerSendMode,
  type ComposerSendPrefs,
  DOUBLE_ENTER_DEFAULT_MS,
  DOUBLE_ENTER_MAX_MS,
  DOUBLE_ENTER_MIN_MS,
  HOLD_MAX_MS,
  HOLD_MIN_MS,
  SEND_GRACE_DEFAULT_MS,
  SEND_GRACE_DEFAULT_REASONS,
  SEND_GRACE_MAX_MS,
  SEND_GRACE_MIN_MS,
  SEND_GRACE_REASONS,
  type SendGraceReason,
  TYPING_IDLE_DEFAULT_MS,
  TYPING_IDLE_MAX_MS,
  TYPING_IDLE_MIN_MS
}

export const $composerSendPrefs = atom<ComposerSendPrefs>(normalizeComposerSendPrefs({ mode: COMPOSER_SEND_DEFAULT_MODE }))

export const $composerSendMode = computed($composerSendPrefs, prefs => prefs.mode)

/** The JSON file main persists to, once we've heard from it. Settings shows it
 *  so the hand-edit path is discoverable rather than folklore. */
export const $composerSendConfigPath = atom<null | string>(null)

/** Whether a bare Enter only breaks the line (every mode except `enter`). */
export const enterBreaksLine = (mode: ComposerSendMode) => mode !== 'enter'

export async function refreshComposerSendPrefs(): Promise<ComposerSendPrefs> {
  const api = window.hermesDesktop?.composerSend

  if (!api) {
    return $composerSendPrefs.get()
  }

  try {
    const result = await api.get()
    const prefs = normalizeComposerSendPrefs(result)
    $composerSendPrefs.set(prefs)
    $composerSendConfigPath.set(result.path || null)

    return prefs
  } catch {
    return $composerSendPrefs.get()
  }
}

export async function setComposerSendPrefs(next: Partial<ComposerSendPrefs>): Promise<ComposerSendPrefs> {
  const requested = normalizeComposerSendPrefs({ ...$composerSendPrefs.get(), ...next })
  const api = window.hermesDesktop?.composerSend

  if (!api) {
    $composerSendPrefs.set(requested)

    return requested
  }

  try {
    const result = await api.set(requested)
    const applied = normalizeComposerSendPrefs(result)
    $composerSendPrefs.set(applied)
    $composerSendConfigPath.set(result.path || null)

    return applied
  } catch (error) {
    // Leave the atom at the last known-good value and surface the failure — an
    // optimistic set here would show a mode that was never persisted and
    // silently reverts on restart.
    notifyError(error, 'Could not save the composer send setting')

    return $composerSendPrefs.get()
  }
}

if (typeof window !== 'undefined' && window.hermesDesktop?.composerSend) {
  void refreshComposerSendPrefs()
}
