/**
 * Window translucency (see-through window).
 *
 * One lever, 0–100. Two modes decide HOW the desktop shows through — see
 * `@hermes/shared/translucency`, which owns the mapping both this store and the
 * main process read.
 *
 * Settings are kept per light/dark appearance: a tint that reads as a whisper
 * over a dark palette is a milky sheet over a light one. The book of settings
 * is the persisted unit (`TranslucencyBook`); `$translucency` publishes the
 * RESOLVED state for the appearance currently painted, so every consumer —
 * the CSS field surfaces, the main process, the HUD — keeps reading one flat
 * state and never has to know appearances were split.
 *
 * The renderer owns the value and mirrors the resolved state to the main
 * process over IPC. Glass additionally needs page-level work, which lives
 * here: the field surfaces have to get out of the way for the platform
 * material underneath the web contents to read (see the `[data-hermes-glass]`
 * block in styles.css).
 */

import {
  type Appearance,
  clampIntensity,
  defaultTranslucencyValues,
  GLASS_MATERIALS,
  GLASS_SCOPES,
  type GlassMaterial,
  glassMaterialForPicker,
  glassMaterialsFor,
  type GlassScope,
  glassSurfaceKeep,
  normalizeBook,
  normalizeMaterial,
  normalizeScope,
  resolveTranslucency,
  setTranslucencyValues,
  TRANSLUCENCY_MAX,
  TRANSLUCENCY_MIN,
  TRANSLUCENCY_STEP,
  type TranslucencyBook,
  type TranslucencyMode,
  type TranslucencyState,
  type TranslucencyValues
} from '@hermes/shared/translucency'
import { atom, computed } from 'nanostores'

import { queryAllVisible } from '@/components/pane-shell/pane-visibility'
import { hostHas } from '@/lib/host-capabilities'
import { isMacPlatform, isWindowsPlatform } from '@/lib/platform'
import { readJson, readKey, writeJson, writeKey } from '@/lib/storage'

export {
  defaultTranslucencyValues,
  GLASS_MATERIALS,
  GLASS_SCOPES,
  glassMaterialForPicker,
  glassMaterialsFor,
  TRANSLUCENCY_MAX,
  TRANSLUCENCY_MIN,
  TRANSLUCENCY_STEP
}

export type { Appearance }

/**
 * Glass needs a native window material. Electron is authoritative (preload
 * sets `hermesDesktop.glassSupported` from `os.release()` so Win10 cannot
 * sneak through). Tests and non-Electron shells fall back to a UA sniff —
 * Mac or Windows — which is why this file pins `navigator.platform` before
 * import.
 */
/** Browser-hosted Desktop has no native window material or opacity surface. */
export const GLASS_SUPPORTED =
  hostHas('windowMaterial') &&
  (typeof window !== 'undefined' && typeof window.hermesDesktop?.glassSupported === 'boolean'
    ? window.hermesDesktop.glassSupported
    : isMacPlatform() || isWindowsPlatform())

/**
 * Whether the setting is offered at all. Linux has no native material and
 * Settings hides the row there, so the setting also resolves to off there
 * (see `$translucency`): a hidden lever must not move the window.
 */
export const TRANSLUCENCY_SUPPORTED =
  hostHas('windowMaterial') &&
  (typeof window !== 'undefined' && typeof window.hermesDesktop?.translucencySupported === 'boolean'
    ? window.hermesDesktop.translucencySupported
    : isMacPlatform() || isWindowsPlatform())

/** Windows collapses the frost ladder — see `glassMaterialsFor`. */
export const GLASS_IS_WINDOWS = GLASS_SUPPORTED && !isMacPlatform()

// v1 held a flat state (one setting for both appearances); v2 is the book.
// Reading v1 as the seed is what carries an already-tuned window across the
// upgrade — normalizeBook lands those values in `base`, which both appearances
// inherit until one of them is edited.
const KEY = 'hermes.desktop.translucency.v2'
const LEGACY_KEY = 'hermes.desktop.translucency.v1'

// An older build persisted the book at every load, so a stored default is no
// choice: keep it provisional, or a browser upgraded from that build would
// seed it into the machine's shared store over native's tuned glass (#145536).
if (typeof window !== 'undefined') {
  const stored = readKey(KEY)

  const book = JSON.stringify(normalizeBook(readJson<unknown>(KEY), true))

  if (stored !== null && [true, false].some(glass => book === JSON.stringify(normalizeBook(null, glass)))) {
    writeKey(KEY, stored, { isDefault: true, provisional: true })
  }
}

const read = (): TranslucencyBook =>
  normalizeBook(readJson<unknown>(KEY) ?? readJson<unknown>(LEGACY_KEY), GLASS_SUPPORTED)

/** The persisted book. Settings edits it; everything else reads `$translucency`. */
export const $translucencyBook = atom<TranslucencyBook>(
  typeof window === 'undefined' ? normalizeBook(null, false) : read()
)

/**
 * Which palette is on screen. Published by the theme provider from its
 * RENDERED mode (background luminance), not the light/dark preference — a
 * skin that keeps a bright surface in "dark" wants light's tint.
 */
export const $appearance = atom<Appearance>('dark')

export function setAppearance(appearance: Appearance): void {
  if ($appearance.get() !== appearance) {
    $appearance.set(appearance)
  }
}

/**
 * The resolved state for the painted appearance — the shape every consumer reads.
 *
 * Where Settings hides the row, the setting resolves to off. The platform
 * defaults (29% tint) would otherwise still resolve there, reach main over IPC
 * and land in translucency.json, and Electron 44 does fade a Linux window when
 * handed that value. Resolving to off also rewrites a value an earlier build
 * persisted, so a stuck profile heals on its first launch.
 */
export const $translucency = computed([$translucencyBook, $appearance], (book, appearance) => {
  const state = resolveTranslucency(book, appearance, isWindowsPlatform())

  return TRANSLUCENCY_SUPPORTED ? state : { ...state, intensity: 0, fade: 0 }
})

/** Write an edit against the appearance being painted. */
const edit = (patch: Partial<TranslucencyValues>): void => {
  $translucencyBook.set(setTranslucencyValues($translucencyBook.get(), $appearance.get(), patch))
}

export function setTranslucency(intensity: number): void {
  edit({ intensity: clampIntensity(intensity) })
}

export function setTranslucencyFade(fade: number): void {
  edit({ fade: clampIntensity(fade) })
}

export function setTranslucencyMode(mode: TranslucencyMode): void {
  $translucencyBook.set({
    ...$translucencyBook.get(),
    mode: mode === 'glass' && GLASS_SUPPORTED ? 'glass' : 'clear'
  })
}

export function setTranslucencyMaterial(material: GlassMaterial): void {
  edit({ material: normalizeMaterial(material) })
}

export function setTranslucencyScope(scope: GlassScope): void {
  edit({ scope: normalizeScope(scope) })
}

// Glass thins surfaces only in real chat windows (the primary window and
// secondary session windows). The HUD, pet overlay, quick entry and wake
// indicator are transparent special-purpose windows that manage their own
// backgrounds — a page-surface rewrite there would fight them.
const CHAT_WINDOW_KINDS = new Set([null, 'secondary', 'browser'])

export const isChatWindow = (search = typeof window === 'undefined' ? '' : window.location.search): boolean => {
  try {
    return CHAT_WINDOW_KINDS.has(new URLSearchParams(search).get('win'))
  } catch {
    return false
  }
}

/* Sidebar scope is a PHYSICAL interval: Sessions can be on either side or
   between other panes, independently of document direction. Sidebar owns the
   post-commit/mount/visibility invalidation; size observers cover live sash
   previews (which intentionally do not commit the layout store). */
const RAIL_SELECTOR = '[data-slot="sidebar"]'
const RAIL_BOUNDS = ['--glass-rail-left', '--glass-rail-right'] as const
let railObserver: null | ResizeObserver = null
let railDirectionObserver: MutationObserver | null = null
let railTargets = new Set<Element>()
let railTrackingOn = false
let railFrame: number | null = null

/** Coalesce Sidebar's committed layout changes, never tint-slider ticks. */
export function invalidateSidebarGlassGeometry(): void {
  if (railTrackingOn && railFrame === null) {
    railFrame = requestAnimationFrame(() => {
      railFrame = null
      measureRailBounds()
    })
  }
}

function trackRailSizes(rails: NodeListOf<HTMLElement>): void {
  const targets = new Set<Element>(rails)

  for (const rail of rails) {
    const host = rail.closest<HTMLElement>('[data-tree-group]')
    const groupId = host?.dataset.treeGroup

    // Keep-alive bodies are CSS-anchored outside the split tree. Observe the
    // corresponding SLOT's split tracks, not the body's stable DOM ancestry.
    const slot = groupId
      ? document.querySelector(`[data-tree-group="${CSS.escape(groupId)}"]:not([data-pane-host])`)
      : rail

    for (let parent = slot?.parentElement; parent; parent = parent.parentElement) {
      if (parent.hasAttribute('data-tree-split')) {
        for (const track of parent.children) {
          targets.add(track)
        }
      }
    }
  }

  for (const target of railTargets) {
    if (!targets.has(target)) {
      railObserver?.unobserve(target)
    }
  }

  for (const target of targets) {
    if (!railTargets.has(target)) {
      railObserver?.observe(target)
    }
  }

  railTargets = targets
}

function measureRailBounds(): void {
  if (!railTrackingOn) {
    return
  }

  const root = document.documentElement
  trackRailSizes(document.querySelectorAll<HTMLElement>(RAIL_SELECTOR))
  let left = 0
  let right = 0

  // Inactive tabs retain their full layout box. Follow the shell's visibility
  // policy before measuring, and ignore collapsed/display:none/empty rails.
  for (const rail of queryAllVisible<HTMLElement>(RAIL_SELECTOR)) {
    if (getComputedStyle(rail).visibility !== 'visible') {
      continue
    }

    const rect = rail.getBoundingClientRect()

    if (rect.width <= 0 || rect.height <= 0) {
      continue
    }

    const start = Math.max(0, Math.min(window.innerWidth, Math.round(rect.left)))
    const end = Math.max(start, Math.min(window.innerWidth, Math.round(rect.right)))

    if (end > start) {
      left = start
      right = end

      break
    }
  }

  for (const [index, value] of [left, right].entries()) {
    const pixels = `${value}px`

    // Root custom-property writes invalidate every consuming surface. Sash
    // observers and duplicate lifecycle notifications must be no-ops at rest.
    if (root.style.getPropertyValue(RAIL_BOUNDS[index]) !== pixels) {
      root.style.setProperty(RAIL_BOUNDS[index], pixels)
    }
  }
}

const startRailTracking = (): void => {
  if (railTrackingOn) {
    return
  }

  railTrackingOn = true
  railObserver = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(invalidateSidebarGlassGeometry)
  // Locale direction can reflow a same-size rail without a tree commit. This
  // watches one root attribute, not the streaming document or our style writes.
  railDirectionObserver = new MutationObserver(invalidateSidebarGlassGeometry)
  railDirectionObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['dir'] })
  window.addEventListener('resize', invalidateSidebarGlassGeometry)
  measureRailBounds()
}

const stopRailTracking = (): void => {
  if (!railTrackingOn) {
    return
  }

  railTrackingOn = false

  if (railFrame !== null) {
    cancelAnimationFrame(railFrame)
    railFrame = null
  }

  railObserver?.disconnect()
  railObserver = null
  railDirectionObserver?.disconnect()
  railDirectionObserver = null
  railTargets.clear()
  window.removeEventListener('resize', invalidateSidebarGlassGeometry)
  RAIL_BOUNDS.forEach(name => document.documentElement.style.removeProperty(name))
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

  // Only an edit is written. The book this window loaded, or adopted from a
  // sibling or the machine's shared preferences, already resolves from
  // storage: writing it back would push this window's default or its
  // normalisation (no glass in a browser) into the shared store over the
  // value the user set in native Desktop (#145213). A v1 state still lands in
  // v2 on first load, since v2 does not resolve to it yet.
  const persist = () => {
    storageTimer = null
    const book = $translucencyBook.get()

    if (JSON.stringify(book) !== JSON.stringify(normalizeBook(readJson<unknown>(KEY), GLASS_SUPPORTED))) {
      writeJson(KEY, book)
    }
  }

  // The RESOLVED state drives paint and IPC — main only ever cares about the
  // appearance on screen. Switching light/dark therefore re-sends, which is
  // exactly right: the window's tint and native opacity change with it.
  $translucency.subscribe(state => {
    applyGlassSurfaces(state)
    window.hermesDesktop?.setTranslucency?.(state)
  })

  // Persistence follows the BOOK, so an appearance switch (which changes the
  // resolved state but not the settings) never schedules a pointless write.
  $translucencyBook.subscribe(() => {
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

    if (JSON.stringify(next) !== JSON.stringify($translucencyBook.get())) {
      $translucencyBook.set(next)
    }
  })

  $translucencyPeek.subscribe(count => {
    if (typeof document === 'undefined') {
      return
    }

    document.documentElement.toggleAttribute(PEEK_ATTR, count > 0)
  })
}
