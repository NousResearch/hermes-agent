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
  DEFAULT_GLASS_MATERIAL,
  DEFAULT_GLASS_SCOPE,
  GLASS_MATERIALS,
  GLASS_SCOPES,
  type GlassMaterial,
  type GlassScope,
  glassSurfaceKeep,
  normalizeMaterial,
  normalizeScope,
  TRANSLUCENCY_MAX,
  TRANSLUCENCY_MIN,
  TRANSLUCENCY_STEP,
  type TranslucencyMode,
  type TranslucencyState
} from '@hermes/shared/translucency'
import { atom } from 'nanostores'

import { isMacPlatform } from '@/lib/platform'
import { readJson, writeJson } from '@/lib/storage'

export {
  DEFAULT_GLASS_MATERIAL,
  DEFAULT_GLASS_SCOPE,
  GLASS_MATERIALS,
  GLASS_SCOPES,
  type GlassMaterial,
  type GlassScope,
  glassSurfaceKeep,
  TRANSLUCENCY_MAX,
  TRANSLUCENCY_MIN,
  TRANSLUCENCY_STEP,
  type TranslucencyMode
}

/** Glass rides on the macOS vibrancy material; other platforms only have Clear. */
export const GLASS_SUPPORTED = isMacPlatform()

const KEY = 'hermes.desktop.translucency.v1'

// The v1 key used to hold a bare intensity (`"23"`). Anything without a mode
// predates glass and always behaved as clear, so it stays clear — a saved
// setting must not change how the window looks on update.
const read = (): TranslucencyState => {
  const stored = readJson<unknown>(KEY)

  const record: Record<string, unknown> =
    stored && typeof stored === 'object' ? (stored as Record<string, unknown>) : { intensity: stored }

  return {
    intensity: clampIntensity(record.intensity),
    mode: record.mode === 'glass' && GLASS_SUPPORTED ? 'glass' : 'clear',
    material: normalizeMaterial(record.material),
    scope: normalizeScope(record.scope)
  }
}

const initial: TranslucencyState =
  typeof window === 'undefined'
    ? { intensity: TRANSLUCENCY_MIN, mode: 'clear', material: DEFAULT_GLASS_MATERIAL, scope: DEFAULT_GLASS_SCOPE }
    : read()

export const $translucency = atom<TranslucencyState>(initial)

export function setTranslucency(intensity: number): void {
  $translucency.set({ ...$translucency.get(), intensity: clampIntensity(intensity) })
}

export function setTranslucencyMode(mode: TranslucencyMode): void {
  $translucency.set({ ...$translucency.get(), mode: mode === 'glass' && GLASS_SUPPORTED ? 'glass' : 'clear' })
}

export function setTranslucencyMaterial(material: GlassMaterial): void {
  $translucency.set({ ...$translucency.get(), material: normalizeMaterial(material) })
}

export function setTranslucencyScope(scope: GlassScope): void {
  $translucency.set({ ...$translucency.get(), scope: normalizeScope(scope) })
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

/* Sidebar scope needs the rail's visual edge published on :root so <body>
   can split its paint there (glass left of the seam, opaque chrome right of
   it — the Finder shape). The rail is an in-flow div whose WIDTH animates
   (components/ui/sidebar.tsx, collapsible='none' branch), so a
   ResizeObserver sees every collapse/expand frame; a window resize listener
   and a re-measure on every store sync cover the rest. RTL flips which side
   the seam is measured from; styles.css picks the matching gradient
   direction off html[dir]. */
let railObserver: null | ResizeObserver = null
let railTarget: Element | null = null
let railTrackingOn = false

const measureRailEdge = (): void => {
  const root = document.documentElement
  const rail = document.querySelector('[data-slot="sidebar"]')

  if (rail !== railTarget) {
    if (railObserver && railTarget) {
      railObserver.unobserve(railTarget)
    }

    railTarget = rail

    if (railObserver && rail) {
      railObserver.observe(rail)
    }
  }

  if (!rail) {
    // No rail in this window (e.g. a pane-only layout): the seam sits at the
    // window edge and the whole field stays opaque — glass simply waits for
    // a rail to exist.
    root.style.setProperty('--glass-rail-edge', '0px')

    return
  }

  const rect = rail.getBoundingClientRect()
  const rtl = getComputedStyle(root).direction === 'rtl'
  const edge = rtl ? window.innerWidth - rect.left : rect.right

  root.style.setProperty('--glass-rail-edge', `${Math.max(0, Math.round(edge))}px`)
}

const startRailTracking = (): void => {
  if (railTrackingOn) {
    measureRailEdge()

    return
  }

  railTrackingOn = true

  if (typeof ResizeObserver !== 'undefined' && !railObserver) {
    railObserver = new ResizeObserver(() => measureRailEdge())
  }

  window.addEventListener('resize', measureRailEdge)
  measureRailEdge()
}

const stopRailTracking = (): void => {
  if (!railTrackingOn) {
    return
  }

  railTrackingOn = false

  if (railObserver && railTarget) {
    railObserver.unobserve(railTarget)
  }

  railTarget = null
  window.removeEventListener('resize', measureRailEdge)
  document.documentElement.style.removeProperty('--glass-rail-edge')
}

/* Peek: while the user is actively adjusting translucency from Settings, the
   overlay they stand in covers the very effect they're tuning — the scrim
   plus a deliberately near-opaque card ([data-glass-raised]). A peek ghosts
   the whole overlay layer so the live window IS the preview (see the
   [data-hermes-translucency-peek] rules in styles.css). A counter rather
   than a boolean: a held slider drag and a timed pulse from a picker click
   can overlap. */
const PEEK_ATTR = 'data-hermes-translucency-peek'

export const $translucencyPeek = atom<number>(0)

export function beginTranslucencyPeek(): void {
  $translucencyPeek.set($translucencyPeek.get() + 1)
}

export function endTranslucencyPeek(): void {
  $translucencyPeek.set(Math.max(0, $translucencyPeek.get() - 1))
}

/**
 * Timed peek for one-shot changes (frost / area / mode clicks, keyboard
 * slider steps): long enough to read the effect, short enough to hand the
 * settings back without feeling stuck.
 */
export function pulseTranslucencyPeek(ms = 900): void {
  beginTranslucencyPeek()

  if (typeof window === 'undefined') {
    endTranslucencyPeek()

    return
  }

  window.setTimeout(endTranslucencyPeek, ms)
}

const applyGlassSurfaces = ({ intensity, mode, scope }: TranslucencyState): void => {
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
    root.setAttribute('data-hermes-glass-scope', scope)
    root.style.setProperty('--translucency-glass-keep', `${glassSurfaceKeep(intensity)}%`)
  } else {
    root.removeAttribute('data-hermes-glass-scope')
    root.style.removeProperty('--translucency-glass-keep')
  }

  if (glassOn && scope === 'sidebar') {
    startRailTracking()
  } else {
    stopRailTracking()
  }
}

if (typeof window !== 'undefined') {
  // The intensity slider fires ~100 updates per drag, and two of the three
  // things a sync does are expensive per-tick: localStorage.setItem is
  // synchronous, and the IPC send wakes the main process. Only the page paint
  // has to keep up with the hand — nothing reads the persisted value until a
  // cold launch, and main's own work is diffed anyway.
  //
  // So: paint every tick, persist and notify on a trailing edge.
  let flushTimer: null | number = null

  const flush = () => {
    flushTimer = null

    const state = $translucency.get()

    writeJson(KEY, state)
    window.hermesDesktop?.setTranslucency?.(state)
  }

  $translucency.subscribe(state => {
    applyGlassSurfaces(state)

    if (flushTimer !== null) {
      window.clearTimeout(flushTimer)
    }

    flushTimer = window.setTimeout(flush, 120)
  })

  // A window closing mid-drag must not lose the setting.
  window.addEventListener('pagehide', () => {
    if (flushTimer !== null) {
      window.clearTimeout(flushTimer)
      flush()
    }
  })

  $translucencyPeek.subscribe(count => {
    if (typeof document === 'undefined') {
      return
    }

    document.documentElement.toggleAttribute(PEEK_ATTR, count > 0)
  })
}
