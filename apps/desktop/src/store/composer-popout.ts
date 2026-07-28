import { atom, computed, type ReadableAtom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'

const POPOUT_STORAGE_KEY = 'hermes.desktop.composerPopout.zones.v1'

// Pre-zone keys: one flag + one position for the whole window. Read at load to
// seed the first zone the user touches (see `legacySeed`), never written again.
const LEGACY_ENABLED_KEY = 'hermes.desktop.composerPopout.enabled'
const LEGACY_POSITION_KEY = 'hermes.desktop.composerPopout.position'

/** Where the floating composer's bottom-right corner sits, measured as an inset
 *  from the viewport's bottom/right edges. Anchoring to the bottom-right keeps
 *  the box visually pinned to its default corner as the window resizes and as
 *  the box grows upward while typing (the corner stays put, height climbs). */
export interface PopoutPosition {
  bottom: number
  right: number
}

/** One layout zone's pop-out state. */
export interface PopoutZoneState {
  poppedOut: boolean
  /** The user's intended placement for this zone, UNCLAMPED — every surface in
   *  the zone renders `clampPopoutPosition` of it against its own rect. */
  position: PopoutPosition
}

// Floating composer width (rem). Shared by the inline style that sets
// --composer-popout-width and the peel-off drag math.
export const POPOUT_WIDTH_REM = 19.5

// Default pop-out placement: tucked into the bottom-right of the thread, clear
// of the window chrome. Matches the brief's "default to the right bottom".
const DEFAULT_POSITION: PopoutPosition = { bottom: 24, right: 24 }

const DEFAULT_ZONE: PopoutZoneState = { poppedOut: false, position: DEFAULT_POSITION }

const isPosition = (value: unknown): value is PopoutPosition => {
  const r = value as null | Partial<PopoutPosition>

  return typeof r?.bottom === 'number' && typeof r?.right === 'number'
}

/** The pre-zone position, if the user had one. */
function legacyPosition(): PopoutPosition {
  try {
    const parsed = JSON.parse(storedString(LEGACY_POSITION_KEY) || 'null') as unknown

    return isPosition(parsed) ? { bottom: parsed.bottom, right: parsed.right } : DEFAULT_POSITION
  } catch {
    return DEFAULT_POSITION
  }
}

function load(): Record<string, PopoutZoneState> {
  const out: Record<string, PopoutZoneState> = {}

  try {
    const parsed = JSON.parse(storedString(POPOUT_STORAGE_KEY) || 'null') as unknown

    for (const [id, value] of Object.entries((parsed as Record<string, unknown>) ?? {})) {
      const zone = value as null | Partial<PopoutZoneState>

      if (typeof zone?.poppedOut === 'boolean' && isPosition(zone.position)) {
        out[id] = { poppedOut: zone.poppedOut, position: { ...zone.position } }
      }
    }
  } catch {
    // Treat unparseable persisted state as missing.
  }

  return out
}

/** Every zone's pop-out state, keyed by layout-tree group id.
 *
 *  A GROUP is one stack of tabs, so this is the scope the user experiences:
 *  tabs in the same zone share a float (switch tabs, the box stays where you
 *  put it), while a split zone beside them keeps its own — popping out on the
 *  left doesn't fling a composer out of the right. */
export const $composerPopoutZones = atom<Record<string, PopoutZoneState>>(load())

/** Write-through to storage. Called explicitly — NOT on every store change: a
 *  drag updates the position once per frame, and serializing every zone to
 *  localStorage at 60Hz is exactly the IO the drag path was built to avoid. */
const persistZones = () => persistString(POPOUT_STORAGE_KEY, JSON.stringify($composerPopoutZones.get()))

/** Whether the user had a float before zones existed. Seeds a zone's first read
 *  so an upgrade doesn't silently dock someone who left their composer floating
 *  — but ONLY until they touch any zone. Once real per-zone state exists, that
 *  is the truth, and a zone split later starts docked like any other. */
let legacySeed: PopoutZoneState | null =
  storedString(LEGACY_ENABLED_KEY) === 'true' ? { poppedOut: true, position: legacyPosition() } : null

const zoneState = (zones: Record<string, PopoutZoneState>, groupId: string): PopoutZoneState =>
  zones[groupId] ?? legacySeed ?? DEFAULT_ZONE

// Cached per-zone derived atoms keep useStore subscriptions referentially stable
// (and keep one zone's drag from re-rendering the composers in another).
const zoneCache = new Map<string, ReadableAtom<PopoutZoneState>>()

export function $composerPopoutZone(groupId: string): ReadableAtom<PopoutZoneState> {
  let cached = zoneCache.get(groupId)

  if (!cached) {
    cached = computed($composerPopoutZones, zones => zoneState(zones, groupId))
    zoneCache.set(groupId, cached)
  }

  return cached
}

export const getComposerPopoutZone = (groupId: string): PopoutZoneState =>
  zoneState($composerPopoutZones.get(), groupId)

export interface PopoutSize {
  height: number
  width: number
}

/** Viewport-space rect the floating composer is confined to. Defaults to the
 *  whole window; pass the thread area so the box can't slide under a pinned
 *  sidebar or behind the header. */
export interface PopoutBounds {
  bottom: number
  left: number
  right: number
  top: number
}

interface SetPositionOptions {
  /** Thread-area rect to confine the box to; falls back to the full window. */
  area?: PopoutBounds
  persist?: boolean
  /** Measured box size; falls back to the compact width + a min height so the
   *  box stays grabbable even when the caller can't measure it. */
  size?: PopoutSize
}

// Keep at least this much between the box and every edge of its bounds, so the
// floating composer can never be dragged (or restored) out of reach.
const EDGE_MARGIN = 8
// Height floor used when the real box height is unknown (init / load / peel-off).
export const POPOUT_ESTIMATED_HEIGHT = 56
const MIN_VISIBLE_HEIGHT = POPOUT_ESTIMATED_HEIGHT

const clampRange = (value: number, lo: number, hi: number) => Math.min(Math.max(value, lo), Math.max(lo, hi))

const rootFontSize = () => parseFloat(getComputedStyle(document.documentElement).fontSize) || 16

/** The chat surface a composer belongs to — the element whose rect bounds its
 *  floating box. Resolved from the composer's OWN surface root, so each mounted
 *  chat surface (primary, session tile, background tab) gets its own area
 *  instead of a document-wide first match. */
export const popoutBoundsElement = (composer: Element | null): Element | null =>
  (composer?.closest('[data-chat-surface]') ?? document).querySelector('[data-slot="composer-bounds"]')

/** The thread area's viewport rect (excludes a pinned sidebar + the header), or
 *  undefined before it mounts — callers then fall back to the full window. */
export function readPopoutBounds(composer: Element | null): PopoutBounds | undefined {
  const el = popoutBoundsElement(composer)

  if (!el) {
    return undefined
  }

  const { bottom, height, left, right, top, width } = el.getBoundingClientRect()

  // Pre-layout (mount before first layout) the rect is empty — fall back to the
  // window rather than clamping the box into a collapsed area.
  return width > 0 && height > 0 ? { bottom, left, right, top } : undefined
}

/**
 * Bound the bottom/right inset so the WHOLE box stays inside `area` (the chat
 * surface's region, or the window by default) — the corner anchor alone would
 * let the box's width/height push it past the opposite edges.
 *
 * PURE and per-surface: a zone stores the user's INTENT, and each mounted
 * composer in it runs that through here against its own rect to get the
 * placement it actually renders. Tabs in a zone are keep-alive mounted, so
 * clamping into the store instead would have every tab overwrite the others
 * with a value bounded by ITS geometry.
 */
export function clampPopoutPosition(
  { bottom, right }: PopoutPosition,
  size?: PopoutSize,
  area?: PopoutBounds
): PopoutPosition {
  const width = size?.width || POPOUT_WIDTH_REM * rootFontSize()
  const height = size?.height || MIN_VISIBLE_HEIGHT
  const { innerHeight: vh, innerWidth: vw } = window
  const a = area ?? { bottom: vh, left: 0, right: vw, top: 0 }

  return {
    bottom: clampRange(bottom, vh - a.bottom + EDGE_MARGIN, vh - a.top - height - EDGE_MARGIN),
    right: clampRange(right, vw - a.right + EDGE_MARGIN, vw - a.left - width - EDGE_MARGIN)
  }
}

function patchZone(groupId: string, patch: Partial<PopoutZoneState>) {
  const zones = $composerPopoutZones.get()
  const stored = zones[groupId]
  const next = { ...zoneState(zones, groupId), ...patch }

  if (
    stored?.poppedOut === next.poppedOut &&
    stored.position.bottom === next.position.bottom &&
    stored.position.right === next.position.right
  ) {
    return
  }

  // The user has now expressed per-zone intent; the pre-zone value stops
  // standing in for zones they haven't touched.
  legacySeed = null
  $composerPopoutZones.set({ ...zones, [groupId]: next })
}

export function setComposerPoppedOut(groupId: string, value: boolean) {
  patchZone(groupId, { poppedOut: value })
  persistZones()
}

/** Move this zone's box. Used per-frame during a drag, so it only writes the
 *  in-memory store by default; pass `persist` for the resting position on
 *  release. Returns the clamped position so callers can sync their live ref. */
export function setComposerPopoutPosition(
  groupId: string,
  position: PopoutPosition,
  { area, persist, size }: SetPositionOptions = {}
): PopoutPosition {
  const next = clampPopoutPosition(position, size, area)
  patchZone(groupId, { position: next })

  if (persist) {
    persistZones()
  }

  return next
}

/** Drop state for zones that no longer exist, so a long-lived install doesn't
 *  accumulate an entry per zone the user ever split and closed. */
export function pruneComposerPopoutZones(liveGroupIds: Iterable<string>) {
  const live = new Set(liveGroupIds)
  const zones = $composerPopoutZones.get()
  const next = Object.fromEntries(Object.entries(zones).filter(([id]) => live.has(id)))

  if (Object.keys(next).length !== Object.keys(zones).length) {
    $composerPopoutZones.set(next)
    persistZones()
  }
}
