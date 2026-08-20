/**
 * Window translucency (see-through window).
 *
 * One lever, 0–100. 0 = off (fully opaque, the default). Two modes decide HOW
 * the desktop shows through — see `@hermes/shared/translucency`, which owns the
 * mapping both this store and the main process read.
 *
 * The renderer owns the value and mirrors it to the main process over IPC.
 * Glass additionally needs page-level work, which lives here: the field
 * surfaces have to get out of the way for the platform material underneath the
 * web contents to read (see the `[data-hermes-glass]` block in styles.css).
 */

import {
  clampIntensity,
  GLASS_MATERIALS,
  GLASS_SCOPES,
  type GlassMaterial,
  glassMaterialForPicker,
  glassMaterialsFor,
  type GlassScope,
  glassSurfaceKeep,
  normalizeMaterial,
  normalizeScope,
  normalizeState,
  TRANSLUCENCY_MAX,
  TRANSLUCENCY_MIN,
  TRANSLUCENCY_STEP,
  type TranslucencyMode,
  type TranslucencyState
} from '@hermes/shared/translucency'
import { atom } from 'nanostores'

import { isMacPlatform, isWindowsPlatform } from '@/lib/platform'
import { readJson, writeJson } from '@/lib/storage'

export {
  GLASS_MATERIALS,
  GLASS_SCOPES,
  glassMaterialForPicker,
  glassMaterialsFor,
  TRANSLUCENCY_MAX,
  TRANSLUCENCY_MIN,
  TRANSLUCENCY_STEP
}

/**
 * Glass needs a native window material. Electron is authoritative (preload
 * sets `hermesDesktop.glassSupported` from `os.release()` so Win10 cannot
 * sneak through). Tests and non-Electron shells fall back to a UA sniff —
 * Mac or Windows — which is why this file pins `navigator.platform` before
 * import.
 */
export const GLASS_SUPPORTED =
  typeof window !== 'undefined' && typeof window.hermesDesktop?.glassSupported === 'boolean'
    ? window.hermesDesktop.glassSupported
    : isMacPlatform() || isWindowsPlatform()

/**
 * Whether the setting is worth showing at all. Linux has neither half —
 * `setOpacity` is a documented no-op and there is no native material — so
 * Settings hides the row rather than offering a lever that does nothing.
 */
export const TRANSLUCENCY_SUPPORTED =
  typeof window !== 'undefined' && typeof window.hermesDesktop?.translucencySupported === 'boolean'
    ? window.hermesDesktop.translucencySupported
    : isMacPlatform() || isWindowsPlatform()

/** Windows collapses the frost ladder — see `glassMaterialsFor`. */
export const GLASS_IS_WINDOWS = GLASS_SUPPORTED && !isMacPlatform()

const KEY = 'hermes.desktop.translucency.v1'

// The v1 key used to hold a bare intensity (`"23"`). Normalization lives in the
// shared module so this and the main process resolve the default the same way —
// including the legacy rule, where a saved NON-ZERO intensity with no mode means
// a profile that has been rendering as clear all along and must keep doing so.
const read = (): TranslucencyState => {
  const stored = readJson<unknown>(KEY)

  return normalizeState(stored && typeof stored === 'object' ? stored : { intensity: stored }, GLASS_SUPPORTED)
}

const initial: TranslucencyState = typeof window === 'undefined' ? normalizeState(null, false) : read()

export const $translucency = atom<TranslucencyState>(initial)

export function setTranslucency(intensity: number): void {
  $translucency.set({ ...$translucency.get(), intensity: clampIntensity(intensity) })
}

export function setTranslucencyFade(fade: number): void {
  $translucency.set({ ...$translucency.get(), fade: clampIntensity(fade) })
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
    // Already tracking: the ResizeObserver on the rail and the window resize
    // listener own every geometry change from here. Re-measuring per store
    // sync would force a layout read (getBoundingClientRect) right after the
    // tint's style write, once per slider tick — write/read thrash on the
    // drag's hot path for a seam that isn't moving. Two exceptions re-acquire:
    // a rail we haven't FOUND yet (scope enabled before the sidebar mounted),
    // and a rail that REMOUNTED (layout reset swaps the element) — the
    // observer sits on the detached node and never fires again. isConnected
    // is a flag read, so the settled hot path stays a single boolean check.
    if (!railTarget || !railTarget.isConnected) {
      measureRailEdge()
    }

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
 * Drop every outstanding hold at once. The settings surface calls this on
 * unmount: a pointer held on the slider when the overlay closes (Escape
 * mid-drag) never delivers its pointerup to the unmounted element, and a
 * counter stuck above zero would leave the peek attribute on <html> —
 * rendering the NEXT settings overlay ghosted at 8% opacity. Outstanding
 * pulse timers still fire endTranslucencyPeek later; the zero floor makes
 * them no-ops.
 */
export function resetTranslucencyPeek(): void {
  $translucencyPeek.set(0)
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
  // Is the user's Glass setting live at all — the same answer in every window.
  const glassLive = mode === 'glass' && intensity > 0 && GLASS_SUPPORTED
  // ...and may THIS window's field surfaces be rewritten for it. Only real
  // chat windows: the HUD, pet overlay, quick entry and wake indicator are
  // transparent windows that own their backgrounds, and the surface rewrite
  // would fight them. The HUD still wants the first answer, because its band
  // paints the app's field mix from `--translucency-glass-keep` and its native
  // frost is gated on the setting being on (see the `[data-hud-glass]` rules
  // and hudFrostFor) — which is why these are two flags and not one.
  const glassOn = glassLive && isChatWindow()
  // Clear mode fades the whole window uniformly, so overlay text and the
  // covered transcript blend; styles.css strengthens the overlay scrim while
  // this attribute is present. Native opacity applies in every window kind, so
  // no chat-window gate.
  const clearOn = mode === 'clear' && intensity > 0

  root.toggleAttribute('data-hermes-glass-on', glassLive)
  root.toggleAttribute('data-hermes-glass', glassOn)
  root.toggleAttribute('data-hermes-clear', clearOn)

  if (glassLive) {
    root.style.setProperty('--translucency-glass-keep', `${glassSurfaceKeep(intensity)}%`)
  } else {
    root.style.removeProperty('--translucency-glass-keep')
  }

  if (glassOn) {
    root.setAttribute('data-hermes-glass-scope', scope)
  } else {
    root.removeAttribute('data-hermes-glass-scope')
  }

  if (glassOn && scope === 'sidebar') {
    startRailTracking()
  } else {
    stopRailTracking()
  }
}

if (typeof window !== 'undefined') {
  // The intensity slider fires ~100 updates per drag. The expensive per-tick
  // work is the synchronous localStorage.setItem — so THAT is debounced.
  // Everything the user can see must track the hand: the page paint (glass is
  // rendered here) AND the IPC send, because in clear mode the effect is the
  // native window opacity and main can only move it when told. Main diffs the
  // state and debounces its own disk write, so per-tick sends cost one
  // setOpacity in clear mode and nothing at all under glass.
  let storageTimer: null | number = null

  const persist = () => {
    storageTimer = null
    writeJson(KEY, $translucency.get())
  }

  $translucency.subscribe(state => {
    applyGlassSurfaces(state)
    window.hermesDesktop?.setTranslucency?.(state)

    if (storageTimer !== null) {
      window.clearTimeout(storageTimer)
    }

    storageTimer = window.setTimeout(persist, 120)
  })

  // A window closing mid-drag must not lose the setting.
  window.addEventListener('pagehide', () => {
    if (storageTimer !== null) {
      window.clearTimeout(storageTimer)
      persist()
    }
  })

  // Cross-window sync (same pattern as themes/context and store/session):
  // under glass an intensity change is painted entirely by each renderer —
  // main deliberately touches nothing native — so a second chat window only
  // learns about it through the storage event its sibling's debounced write
  // fires. Without this, window B's tint freezes until reload.
  window.addEventListener('storage', event => {
    if (event.key !== KEY) {
      return
    }

    const next = read()
    const current = $translucency.get()

    if (
      next.intensity !== current.intensity ||
      next.fade !== current.fade ||
      next.mode !== current.mode ||
      next.material !== current.material ||
      next.scope !== current.scope
    ) {
      $translucency.set(next)
    }
  })

  $translucencyPeek.subscribe(count => {
    if (typeof document === 'undefined') {
      return
    }

    document.documentElement.toggleAttribute(PEEK_ATTR, count > 0)
  })
}
