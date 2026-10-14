import { computed, type ReadableAtom } from 'nanostores'

import { LAYOUT_KEYS } from '@/lib/layout-persistence'
import { Codecs } from '@/lib/persisted'
import { modeLayout } from '@/store/interface-mode'

export interface PaneStateSnapshot {
  open: boolean
  widthOverride?: number
  /** Vertical size override (px) for panes that resize on the Y axis (e.g. the bottom-row terminal). */
  heightOverride?: number
}

export interface PaneRegisterDefaults {
  open: boolean
  widthOverride?: number
}

function isSnapshot(value: unknown): value is PaneStateSnapshot {
  if (!value || typeof value !== 'object') {
    return false
  }

  const r = value as Record<string, unknown>

  if (typeof r.open !== 'boolean') {
    return false
  }

  const widthOk =
    r.widthOverride === undefined || (typeof r.widthOverride === 'number' && Number.isFinite(r.widthOverride))

  const heightOk =
    r.heightOverride === undefined || (typeof r.heightOverride === 'number' && Number.isFinite(r.heightOverride))

  return widthOk && heightOk
}

const paneDefaults: Record<string, PaneStateSnapshot> = {}

export const $paneStates = modeLayout.atom<Record<string, PaneStateSnapshot>>(
  LAYOUT_KEYS.panes,
  () => ({ ...paneDefaults }),
  Codecs.json(parsed => {
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return { ...paneDefaults }
    }

    return { ...paneDefaults, ...Object.fromEntries(Object.entries(parsed).filter(([, value]) => isSnapshot(value))) }
  })
)

// Cached per-pane derived atoms keep useStore subscriptions referentially stable.
function memoized<T>(
  cache: Map<string, ReadableAtom<T>>,
  id: string,
  selector: (s: PaneStateSnapshot | undefined) => T
) {
  let cached = cache.get(id)

  if (!cached) {
    cached = computed($paneStates, states => selector(states[id]))
    cache.set(id, cached)
  }

  return cached
}

const openCache = new Map<string, ReadableAtom<boolean>>()
const stateCache = new Map<string, ReadableAtom<PaneStateSnapshot | undefined>>()
const widthCache = new Map<string, ReadableAtom<number | undefined>>()
const heightCache = new Map<string, ReadableAtom<number | undefined>>()

export const $paneOpen = (id: string) => memoized(openCache, id, s => s?.open ?? false)
export const $paneState = (id: string) => memoized(stateCache, id, s => s)
export const $paneWidthOverride = (id: string) => memoized(widthCache, id, s => s?.widthOverride)
export const $paneHeightOverride = (id: string) => memoized(heightCache, id, s => s?.heightOverride)

export function ensurePaneRegistered(id: string, defaults: PaneRegisterDefaults) {
  paneDefaults[id] = { ...defaults }
  const current = $paneStates.get()

  if (current[id] !== undefined) {
    return
  }

  $paneStates.set({ ...current, [id]: { open: defaults.open, widthOverride: defaults.widthOverride } })
}

export function setPaneOpen(id: string, open: boolean) {
  const current = $paneStates.get()
  const existing = current[id]

  if (existing?.open === open) {
    return
  }

  $paneStates.set({ ...current, [id]: { ...existing, open } })
}

export function togglePane(id: string) {
  const current = $paneStates.get()
  const existing = current[id]
  $paneStates.set({ ...current, [id]: { ...existing, open: !(existing?.open ?? false) } })
}

/**
 * The state an override write starts from. A sash drag (and a double-click on
 * a sash) reaches EVERY shown pane of the zone, including one no other surface
 * ever registered — and inventing `open: false` there CLOSED the pane the user
 * was looking at, which then re-divided the heights of everything stacked
 * beside it (a hidden zone stops rendering, so its column re-splits). Openness
 * belongs to `setPaneOpen`/`togglePane` alone: an override starts from the
 * pane's own registered default, and a pane with no default at all is one that
 * is on screen — a zone is only draggable while visible — so it is open.
 */
function overrideBase(states: Record<string, PaneStateSnapshot>, id: string): PaneStateSnapshot {
  return states[id] ?? paneDefaults[id] ?? { open: true }
}

export function setPaneWidthOverride(id: string, width: number | undefined) {
  const current = $paneStates.get()
  const existing = overrideBase(current, id)

  if (existing.widthOverride === width) {
    return
  }

  $paneStates.set({ ...current, [id]: { ...existing, widthOverride: width } })
}

export function setPaneHeightOverride(id: string, height: number | undefined) {
  const current = $paneStates.get()
  const existing = overrideBase(current, id)

  if (existing.heightOverride === height) {
    return
  }

  $paneStates.set({ ...current, [id]: { ...existing, heightOverride: height } })
}

export const clearPaneWidthOverride = (id: string) => setPaneWidthOverride(id, undefined)
export const clearPaneHeightOverride = (id: string) => setPaneHeightOverride(id, undefined)

/** Drop every pane's drag-resize override (open state untouched). Layout
 *  reset / preset application: zones return to their declared sizes. */
export function clearAllPaneSizeOverrides() {
  const current = $paneStates.get()
  let changed = false
  const next: Record<string, PaneStateSnapshot> = {}

  for (const [id, state] of Object.entries(current)) {
    if (state.widthOverride !== undefined || state.heightOverride !== undefined) {
      changed = true
      next[id] = { open: state.open }
    } else {
      next[id] = state
    }
  }

  if (changed) {
    $paneStates.set(next)
  }
}

export const getPaneStateSnapshot = (id: string) => $paneStates.get()[id]
