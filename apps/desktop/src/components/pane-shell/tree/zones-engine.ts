/**
 * FancyZones runtime engine — verbatim port of the drag/highlight internals
 * from PowerToys `FancyZonesLib` (microsoft/PowerToys, MIT):
 *
 *  - `Layout.cpp`   -> `zonesFromPoint` (sensitivity-radius capture, the
 *    "captured but not strictly captured" rejection, the overlap resolution
 *    algorithms incl. ClosestCenter with OVERLAPPING_CENTERS_SENSITIVITY),
 *    `getCombinedZoneRange`, `getCombinedZonesRect`.
 *  - `HighlightedZones.cpp` -> the `HighlightedZones` update state machine
 *    (initial-zone latch + combined range while "select many" is held).
 *  - `ZonesOverlay.cpp` -> animation timing: FadeInDurationMillis = 200,
 *    FlashZonesDurationMillis = 700, alpha = clamp(t/200, 0.001, 1), and the
 *    fill-alpha rule (highlightOpacity applies to BOTH inactive + highlight).
 *  - `Colors.cpp` -> ZoneColors resolution (system-theme mode maps accent to
 *    border+highlight and background to primary; number color auto-contrasts).
 */

// ---------------------------------------------------------------------------
// Constants (ZonesOverlay.cpp / Layout.cpp / FancyZones defaults)
// ---------------------------------------------------------------------------

export const FADE_IN_DURATION_MILLIS = 200
export const FLASH_ZONES_DURATION_MILLIS = 700
/** LayoutDefaultSettings::DefaultSensitivityRadius. */
export const DEFAULT_SENSITIVITY_RADIUS = 20
/** ZoneSelectionAlgorithms::OVERLAPPING_CENTERS_SENSITIVITY. */
const OVERLAPPING_CENTERS_SENSITIVITY = 75

export type OverlappingZonesAlgorithm = 'Smallest' | 'Largest' | 'Positional' | 'ClosestCenter'

export interface ZoneRect {
  left: number
  top: number
  right: number
  bottom: number
}

export interface EngineZone {
  id: string
  rect: ZoneRect
}

interface Point {
  x: number
  y: number
}

const zoneArea = (z: EngineZone) => Math.max(0, z.rect.right - z.rect.left) * Math.max(0, z.rect.bottom - z.rect.top)

// ---------------------------------------------------------------------------
// ZonesOverlay::GetAnimationAlpha (verbatim ramp)
// ---------------------------------------------------------------------------

export function getAnimationAlpha(startedAtMs: number, nowMs: number, autoHide: boolean): number {
  const millis = nowMs - startedAtMs

  if (autoHide && millis > FLASH_ZONES_DURATION_MILLIS) {
    return 0
  }

  // Return a positive value to avoid hiding.
  return Math.min(Math.max(millis / FADE_IN_DURATION_MILLIS, 0.001), 1)
}

// ---------------------------------------------------------------------------
// Colors (Colors.cpp + FancyZones settings defaults)
// ---------------------------------------------------------------------------

export interface ZoneColors {
  primaryColor: string
  borderColor: string
  highlightColor: string
  numberColor: string
  /** 0..100, applied as fill alpha to BOTH primary and highlight fills. */
  highlightOpacity: number
}

/** FancyZonesSettings defaults (custom-color mode). */
export const FANCYZONES_DEFAULT_COLORS: ZoneColors = {
  primaryColor: '#F5FCFF',
  borderColor: '#FFFFFF',
  highlightColor: '#008CFF',
  numberColor: '#000000',
  highlightOpacity: 50
}

/**
 * Colors::GetZoneColors systemTheme branch, mapped to our design tokens:
 * accent -> border + highlight, surface -> primary, number auto-contrast
 * (CSS light-dark handles what the C++ does with the black-background check).
 */
export function systemThemeZoneColors(): ZoneColors {
  return {
    primaryColor: 'var(--ui-bg-editor)',
    borderColor: 'var(--ui-accent)',
    highlightColor: 'var(--ui-accent)',
    numberColor: 'var(--ui-text-primary)',
    highlightOpacity: 50
  }
}

// ---------------------------------------------------------------------------
// ZoneSelectionAlgorithms (Layout.cpp, verbatim)
// ---------------------------------------------------------------------------

function zoneSelectPriority(
  zones: Map<string, EngineZone>,
  capturedZones: string[],
  compare: (a: EngineZone, b: EngineZone) => boolean
): string[] {
  let chosen = 0

  for (let i = 1; i < capturedZones.length; i++) {
    if (compare(zones.get(capturedZones[i])!, zones.get(capturedZones[chosen])!)) {
      chosen = i
    }
  }

  return [capturedZones[chosen]]
}

function zoneSelectSubregion(
  zones: Map<string, EngineZone>,
  capturedZones: string[],
  pt: Point,
  sensitivityRadius: number
): string[] {
  const expand = (rect: ZoneRect): ZoneRect => ({
    top: rect.top - sensitivityRadius / 2,
    bottom: rect.bottom + sensitivityRadius / 2,
    left: rect.left - sensitivityRadius / 2,
    right: rect.right + sensitivityRadius / 2
  })

  // Compute the overlapped rectangle.
  let overlap = expand(zones.get(capturedZones[0])!.rect)

  for (let i = 1; i < capturedZones.length; i++) {
    const current = expand(zones.get(capturedZones[i])!.rect)
    overlap = {
      top: Math.max(overlap.top, current.top),
      left: Math.max(overlap.left, current.left),
      bottom: Math.min(overlap.bottom, current.bottom),
      right: Math.min(overlap.right, current.right)
    }
  }

  // Avoid division by zero.
  const width = Math.max(overlap.right - overlap.left, 1)
  const height = Math.max(overlap.bottom - overlap.top, 1)

  const verticalSplit = height > width
  let zoneIndex: number

  if (verticalSplit) {
    zoneIndex = Math.floor(((pt.y - overlap.top) * capturedZones.length) / height)
  } else {
    zoneIndex = Math.floor(((pt.x - overlap.left) * capturedZones.length) / width)
  }

  zoneIndex = Math.min(Math.max(zoneIndex, 0), capturedZones.length - 1)

  return [capturedZones[zoneIndex]]
}

function zoneSelectClosestCenter(zones: Map<string, EngineZone>, capturedZones: string[], pt: Point): string[] {
  const getCenter = (zone: EngineZone): Point => ({
    x: (zone.rect.right + zone.rect.left) / 2,
    y: (zone.rect.top + zone.rect.bottom) / 2
  })

  const pointDifference = (a: Point, b: Point) => (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)
  const distanceFromCenter = (zone: EngineZone) => pointDifference(getCenter(zone), pt)

  const closerToCenter = (zone1: EngineZone, zone2: EngineZone) => {
    if (pointDifference(getCenter(zone1), getCenter(zone2)) > OVERLAPPING_CENTERS_SENSITIVITY) {
      return distanceFromCenter(zone1) < distanceFromCenter(zone2)
    }

    return zoneArea(zone1) < zoneArea(zone2)
  }

  return zoneSelectPriority(zones, capturedZones, closerToCenter)
}

// ---------------------------------------------------------------------------
// Layout::ZonesFromPoint (verbatim)
// ---------------------------------------------------------------------------

export function zonesFromPoint(
  zoneList: EngineZone[],
  pt: Point,
  sensitivityRadius = DEFAULT_SENSITIVITY_RADIUS,
  overlappingAlgorithm: OverlappingZonesAlgorithm = 'Smallest'
): string[] {
  const zones = new Map(zoneList.map(z => [z.id, z]))
  const capturedZones: string[] = []
  const strictlyCapturedZones: string[] = []

  for (const zone of zoneList) {
    const zoneRect = zone.rect

    if (
      zoneRect.left - sensitivityRadius <= pt.x &&
      pt.x <= zoneRect.right + sensitivityRadius &&
      zoneRect.top - sensitivityRadius <= pt.y &&
      pt.y <= zoneRect.bottom + sensitivityRadius
    ) {
      capturedZones.push(zone.id)
    }

    if (zoneRect.left <= pt.x && pt.x < zoneRect.right && zoneRect.top <= pt.y && pt.y < zoneRect.bottom) {
      strictlyCapturedZones.push(zone.id)
    }
  }

  // If only one zone is captured, but it's not strictly captured
  // don't consider it as captured.
  if (capturedZones.length === 1 && strictlyCapturedZones.length === 0) {
    return []
  }

  // If captured zones do not overlap, return all of them.
  // Otherwise, return one of them based on the chosen selection algorithm.
  let overlap = false

  outer: for (let i = 0; i < capturedZones.length; i++) {
    for (let j = i + 1; j < capturedZones.length; j++) {
      const rectI = zones.get(capturedZones[i])!.rect
      const rectJ = zones.get(capturedZones[j])!.rect

      if (
        Math.max(rectI.top, rectJ.top) + sensitivityRadius < Math.min(rectI.bottom, rectJ.bottom) &&
        Math.max(rectI.left, rectJ.left) + sensitivityRadius < Math.min(rectI.right, rectJ.right)
      ) {
        overlap = true

        break outer
      }
    }
  }

  if (overlap) {
    switch (overlappingAlgorithm) {
      case 'Smallest':
        return zoneSelectPriority(zones, capturedZones, (a, b) => zoneArea(a) < zoneArea(b))

      case 'Largest':
        return zoneSelectPriority(zones, capturedZones, (a, b) => zoneArea(a) > zoneArea(b))

      case 'Positional':
        return zoneSelectSubregion(zones, capturedZones, pt, sensitivityRadius)

      case 'ClosestCenter':
        return zoneSelectClosestCenter(zones, capturedZones, pt)
    }
  }

  return capturedZones
}

/** The single zone a drop lands in: ClosestCenter among the highlighted set. */
export function primaryZone(zoneList: EngineZone[], highlighted: string[], pt: Point): string | null {
  if (highlighted.length === 0) {
    return null
  }

  if (highlighted.length === 1) {
    return highlighted[0]
  }

  const zones = new Map(zoneList.map(z => [z.id, z]))

  return zoneSelectClosestCenter(zones, highlighted, pt)[0] ?? highlighted[0]
}

// ---------------------------------------------------------------------------
// Layout::GetCombinedZoneRange (verbatim)
// ---------------------------------------------------------------------------

export function getCombinedZoneRange(zoneList: EngineZone[], initialZones: string[], finalZones: string[]): string[] {
  const zones = new Map(zoneList.map(z => [z.id, z]))
  const combinedZones = [...new Set([...initialZones, ...finalZones])]

  let boundingRect: ZoneRect | null = null

  for (const zoneId of combinedZones) {
    const zone = zones.get(zoneId)

    if (zone) {
      const rect = zone.rect

      if (!boundingRect) {
        boundingRect = { ...rect }
      } else {
        boundingRect.left = Math.min(boundingRect.left, rect.left)
        boundingRect.top = Math.min(boundingRect.top, rect.top)
        boundingRect.right = Math.max(boundingRect.right, rect.right)
        boundingRect.bottom = Math.max(boundingRect.bottom, rect.bottom)
      }
    }
  }

  const result: string[] = []

  if (boundingRect) {
    for (const zone of zoneList) {
      const rect = zone.rect

      if (
        boundingRect.left <= rect.left &&
        rect.right <= boundingRect.right &&
        boundingRect.top <= rect.top &&
        rect.bottom <= boundingRect.bottom
      ) {
        result.push(zone.id)
      }
    }
  }

  return result
}

// ---------------------------------------------------------------------------
// HighlightedZones (HighlightedZones.cpp, verbatim state machine)
// ---------------------------------------------------------------------------

export class HighlightedZones {
  private highlightZone: string[] = []
  private initialHighlightZone: string[] = []

  zones(): readonly string[] {
    return this.highlightZone
  }

  empty(): boolean {
    return this.highlightZone.length === 0
  }

  /** Returns true when the highlight set changed. */
  update(zoneList: EngineZone[], point: Point, selectManyZones: boolean): boolean {
    let highlightZone = zonesFromPoint(zoneList, point)

    if (selectManyZones) {
      if (this.initialHighlightZone.length === 0) {
        // First time.
        this.initialHighlightZone = highlightZone
      } else {
        highlightZone = getCombinedZoneRange(zoneList, this.initialHighlightZone, highlightZone)
      }
    } else {
      this.initialHighlightZone = []
    }

    const updated =
      highlightZone.length !== this.highlightZone.length || highlightZone.some((z, i) => z !== this.highlightZone[i])

    this.highlightZone = highlightZone

    return updated
  }

  reset(): void {
    this.highlightZone = []
    this.initialHighlightZone = []
  }
}
