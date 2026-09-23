import { atom, computed, type ReadableAtom } from 'nanostores'

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

const STORAGE_KEY = 'hermes.desktop.paneStates.v1'

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

function load(): Record<string, PaneStateSnapshot> {
  if (typeof window === 'undefined') {
    return {}
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)

    if (raw) {
      const parsed = JSON.parse(raw) as unknown

      if (parsed && typeof parsed === 'object') {
        const out: Record<string, PaneStateSnapshot> = {}

        for (const [id, value] of Object.entries(parsed as Record<string, unknown>)) {
          if (isSnapshot(value)) {
            out[id] = { open: value.open, widthOverride: value.widthOverride, heightOverride: value.heightOverride }
          }
        }

        return out
      }
    }
  } catch {
    // Treat unparseable persisted state as missing.
  }

  return {}
}

// Persists both open state and resize width; load() validates each snapshot.
function persist(states: Record<string, PaneStateSnapshot>) {
  if (typeof window === 'undefined') {
    return
  }

  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(states))
  } catch {
    // Storage failures are nonfatal.
  }
}

export const $paneStates = atom<Record<string, PaneStateSnapshot>>(load())

$paneStates.subscribe(persist)

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

export function setPaneWidthOverride(id: string, width: number | undefined) {
  const current = $paneStates.get()
  const existing = current[id] ?? { open: false }

  if (existing.widthOverride === width) {
    return
  }

  $paneStates.set({ ...current, [id]: { ...existing, widthOverride: width } })
}

export function setPaneHeightOverride(id: string, height: number | undefined) {
  const current = $paneStates.get()
  const existing = current[id] ?? { open: false }

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
