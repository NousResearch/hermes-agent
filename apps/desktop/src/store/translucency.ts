/**
 * Window translucency (see-through window).
 *
 * One lever, 0–100. 0 = off (fully opaque, the default). Two modes decide HOW
 * the desktop shows through — see `@hermes/shared/translucency`, which owns the
 * mapping both this store and the main process read.
 *
 * The renderer owns the value and mirrors it to the main process over IPC.
 * Glass additionally needs page-level work, which lives here: the field
 * surfaces have to get out of the way for the vibrancy material underneath the
 * web contents to read (see the `[data-hermes-glass]` block in styles.css).
 */

import {
  clampIntensity,
  glassSurfaceKeep,
  TRANSLUCENCY_MAX,
  TRANSLUCENCY_MIN,
  TRANSLUCENCY_STEP,
  type TranslucencyMode
} from '@hermes/shared/translucency'
import { atom } from 'nanostores'

import { isMacPlatform } from '@/lib/platform'
import { readJson, writeJson } from '@/lib/storage'

export { TRANSLUCENCY_MAX, TRANSLUCENCY_MIN, TRANSLUCENCY_STEP, type TranslucencyMode }

/** Glass rides on the macOS vibrancy material; other platforms only have Clear. */
export const GLASS_SUPPORTED = isMacPlatform()

const KEY = 'hermes.desktop.translucency.v1'

interface PersistedTranslucency {
  intensity: number
  mode: TranslucencyMode
}

// The v1 key used to hold a bare intensity (`"23"`). Anything without a mode
// predates glass and always behaved as clear, so it stays clear — a saved
// setting must not change how the window looks on update.
const read = (): PersistedTranslucency => {
  const stored = readJson<unknown>(KEY)

  const record: Record<string, unknown> =
    stored && typeof stored === 'object' ? (stored as Record<string, unknown>) : { intensity: stored }

  return {
    intensity: clampIntensity(record.intensity),
    mode: record.mode === 'glass' && GLASS_SUPPORTED ? 'glass' : 'clear'
  }
}

const initial: PersistedTranslucency =
  typeof window === 'undefined' ? { intensity: TRANSLUCENCY_MIN, mode: 'clear' } : read()

export const $translucency = atom<PersistedTranslucency>(initial)

export function setTranslucency(intensity: number): void {
  $translucency.set({ ...$translucency.get(), intensity: clampIntensity(intensity) })
}

export function setTranslucencyMode(mode: TranslucencyMode): void {
  $translucency.set({ ...$translucency.get(), mode: mode === 'glass' && GLASS_SUPPORTED ? 'glass' : 'clear' })
}

// Glass thins surfaces only in real chat windows (the primary window and
// secondary session windows). The HUD, pet overlay, quick entry and wake
// indicator are transparent special-purpose windows that manage their own
// backgrounds — a page-surface rewrite there would fight them.
const CHAT_WINDOW_KINDS = new Set([null, 'secondary'])

export const isChatWindow = (search = typeof window === 'undefined' ? '' : window.location.search): boolean => {
  try {
    return CHAT_WINDOW_KINDS.has(new URLSearchParams(search).get('win'))
  } catch {
    return false
  }
}

const applyGlassSurfaces = ({ intensity, mode }: PersistedTranslucency): void => {
  if (typeof document === 'undefined') {
    return
  }

  const root = document.documentElement
  const glassOn = mode === 'glass' && intensity > 0 && GLASS_SUPPORTED && isChatWindow()
  // Clear mode fades the whole window uniformly, so overlay text and the
  // covered transcript blend; styles.css strengthens the overlay scrim while
  // this attribute is present. Native opacity applies in every window kind, so
  // no chat-window gate.
  const clearOn = mode === 'clear' && intensity > 0

  root.toggleAttribute('data-hermes-glass', glassOn)
  root.toggleAttribute('data-hermes-clear', clearOn)

  if (glassOn) {
    root.style.setProperty('--translucency-glass-keep', `${glassSurfaceKeep(intensity)}%`)
  } else {
    root.style.removeProperty('--translucency-glass-keep')
  }
}

if (typeof window !== 'undefined') {
  $translucency.subscribe(state => {
    writeJson(KEY, state)
    applyGlassSurfaces(state)
    window.hermesDesktop?.setTranslucency?.(state)
  })
}
