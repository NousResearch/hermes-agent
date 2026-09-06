import { atom, computed, type ReadableAtom } from 'nanostores'

export interface PaneStateSnapshot {
  open: boolean
  widthOverride?: number
  /** Vertical size override (px) for panes that resize on the Y axis (e.g. the bottom-row terminal). */
  heightOverride?: number
  /** When true the pane's width is locked: the track model uses `lockedWidth`
   *  (or the current override) as a fixed px basis, and sash drags skip this
   *  pane as a donor/receiver. Absent = off (default). */
  lockWidth?: boolean
  /** When true the pane's height is locked along the column axis. */
  lockHeight?: boolean
  /** The width (px) captured at lock time — used as the locked basis. */
  lockedWidth?: number
  /** The height (px) captured at lock time — used as the locked basis. */
  lockedHeight?: number
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

  const numOrUndef = (v: unknown) => v === undefined || (typeof v === 'number' && Number.isFinite(v) && v > 0)

  return (
    numOrUndef(r.widthOverride) &&
    numOrUndef(r.heightOverride) &&
    numOrUndef(r.lockedWidth) &&
    numOrUndef(r.lockedHeight)
  )
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

/** Lock or unlock a pane's width axis. When locking, captures the width:
 *  explicit `fixedWidth` (measured DOM px) wins over the current override,
 *  which wins over undefined (the track model falls back to declared CSS).
 *  Non-positive, non-finite, or NaN values are rejected (no capture).
 *  When unlocking, clears both flags and the captured value. */
export function setPaneWidthLock(id: string, locked: boolean, fixedWidth?: number) {
  const current = $paneStates.get()
  const existing = current[id] ?? { open: false }

  if (locked) {
    if (existing.lockWidth) {
      return
    }

    const captured =
      fixedWidth !== undefined && Number.isFinite(fixedWidth) && fixedWidth > 0 ? fixedWidth : existing.widthOverride

    $paneStates.set({
      ...current,
      [id]: {
        ...existing,
        lockWidth: true as const,
        lockedWidth: captured
      }
    })
  } else {
    if (!existing.lockWidth) {
      return
    }

    const { lockWidth: _lw, lockedWidth: _lwv, ...rest } = existing
    $paneStates.set({ ...current, [id]: rest })
  }
}

/** Lock or unlock a pane's height axis. When locking, captures the height:
 *  explicit `fixedHeight` (measured DOM px) wins over the current override,
 *  which wins over undefined. Non-positive, non-finite, or NaN values are
 *  rejected (no capture). When unlocking, clears both. */
export function setPaneHeightLock(id: string, locked: boolean, fixedHeight?: number) {
  const current = $paneStates.get()
  const existing = current[id] ?? { open: false }

  if (locked) {
    if (existing.lockHeight) {
      return
    }

    const captured =
      fixedHeight !== undefined && Number.isFinite(fixedHeight) && fixedHeight > 0
        ? fixedHeight
        : existing.heightOverride

    $paneStates.set({
      ...current,
      [id]: {
        ...existing,
        lockHeight: true as const,
        lockedHeight: captured
      }
    })
  } else {
    if (!existing.lockHeight) {
      return
    }

    const { lockHeight: _lh, lockedHeight: _lhv, ...rest } = existing
    $paneStates.set({ ...current, [id]: rest })
  }
}

/** Drop every pane's drag-resize override and axis locks (open state
 *  untouched). Layout reset / preset application: zones return to their
 *  declared sizes and unlocked state. */
export function clearAllPaneSizeOverrides() {
  const current = $paneStates.get()
  let changed = false
  const next: Record<string, PaneStateSnapshot> = {}

  for (const [id, state] of Object.entries(current)) {
    const hasLock = state.lockWidth || state.lockHeight

    if (state.widthOverride !== undefined || state.heightOverride !== undefined || hasLock) {
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
