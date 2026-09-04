import { $petMotion, $petRoamDir } from '@/store/pet'

import { dwellMs, pickStrollTarget } from './roam-behavior'
import { groundTop, type Ledge, resolveLedge } from './roam-geometry'

export interface OverlayBounds {
  height: number
  width: number
  x: number
  y: number
}

export interface OverlayRoamEnvironment {
  sceneRevision?: string
  visualLedges?: Ledge[]
  windows: OverlayBounds[]
  workArea: OverlayBounds
}

export interface OverlayRoamOptions {
  enabled: boolean
  isInteracting: () => boolean
  loopMs: number
  petH: number
  petW: number
  /** Changing this restarts planning immediately (for example after a drag). */
  replanKey?: number
}

interface Span {
  left: number
  right: number
}

const PET_FOOT_INSET_PX = 24
const MIN_TRAVEL_PX = 48
const GRAVITY_PX_S2 = 1500
const MAX_DT_S = 0.05
const VERTICAL_ACTION_CHANCE = 0.5
const MIN_HOP_CHANCE = 0.25
const BASE_HOP_DURATION_MS = 800
const LANDING_ANALYSIS_DELAY_MS = 250
const PAUSE_POLL_MS = 250
const SUPPORT_SNAP_TOLERANCE_PX = 8
const DRAG_SETTLE_TOLERANCE_PX = 24
const LEDGE_MATCH_TOLERANCE_PX = 8
const SUPPORT_RETRY_MS = 1000
const SUPPORT_FAILURE_LIMIT = 3
const PLANNED_HOP_FAILURE_LIMIT = 2
const OVERLAY_PAUSE_DWELL = { maxMs: 2500, meanMs: 900, minMs: 250 }

type Phase = 'fall' | 'hop' | 'walk'
export type OverlayIdleAction = 'drop' | 'hop' | 'walk'
export interface OverlayVisualCandidate {
  hits: number
  ledge: Ledge
}

export interface OverlayHopDestination {
  ledge: Ledge
}

export type OverlayVerticalCorrection = 'fall' | 'hop' | 'snap'

/** Search no farther than three rendered pet heights above the pet. */
export const overlayHopMaxVerticalTravel = (petH: number): number => petH * 3

/**
 * Find upper supports directly above the pet's current horizontal position.
 * With no upper surface, hop in place on the current support.
 */
export function overlayHopDestinations(
  ledges: Ledge[],
  fromLedge: Ledge,
  fromX: number,
  petH: number
): OverlayHopDestination[] {
  const verticalReach = overlayHopMaxVerticalTravel(petH)

  const upperSupports = ledges
    .filter(
      ledge =>
        ledge !== fromLedge &&
        ledge.y < fromLedge.y - 1 &&
        fromLedge.y - ledge.y <= verticalReach &&
        fromX >= ledge.left - 2 &&
        fromX <= ledge.right + 2
    )
    .map(ledge => ({ ledge }))

  return upperSupports.length > 0 ? upperSupports : [{ ledge: fromLedge }]
}

/** Every support below the pet's current X is a valid drop destination. */
export function overlayDropDestinations(
  ledges: Ledge[],
  fromLedge: Ledge,
  fromX: number
): OverlayHopDestination[] {
  return ledges
    .filter(
      ledge =>
        ledge !== fromLedge &&
        ledge.y > fromLedge.y + 1 &&
        fromX >= ledge.left - 2 &&
        fromX <= ledge.right + 2
    )
    .map(ledge => ({ ledge }))
}

export function randomOverlayDropDestination(
  candidates: OverlayHopDestination[],
  rng: () => number = Math.random
): OverlayHopDestination | null {
  if (candidates.length === 0) {
    return null
  }

  const sample = Math.max(0, Math.min(1, rng()))
  const index = Math.min(candidates.length - 1, Math.floor(sample * candidates.length))

  return candidates[index]!
}

/** Prefer the nearest valid support above the pet. */
export function nearestOverlayHopDestination(
  candidates: OverlayHopDestination[],
  fromY: number
): OverlayHopDestination | null {
  let nearest: OverlayHopDestination | null = null
  let nearestDistance = Number.POSITIVE_INFINITY

  for (const candidate of candidates) {
    const distance = Math.abs(candidate.ledge.y - fromY)

    if (distance < nearestDistance) {
      nearest = candidate
      nearestDistance = distance
    }
  }

  return nearest
}

/** Rise half a rendered sprite above the selected landing height. */
export function overlayHopApexY(petH: number, targetY: number): number {
  return targetY - petH * 0.5
}

/** A smooth two-part parabola that passes through start, apex, and destination. */
export function overlayHopYAtProgress(startY: number, apexY: number, targetY: number, progress: number): number {
  const t = Math.max(0, Math.min(1, progress))

  if (t <= 0.5) {
    const u = t * 2
    const eased = 1 - (1 - u) ** 2

    return startY + (apexY - startY) * eased
  }

  const u = (t - 0.5) * 2

  return apexY + (targetY - apexY) * u ** 2
}

const clipTo = (bounds: OverlayBounds, area: OverlayBounds): OverlayBounds | null => {
  const x = Math.max(bounds.x, area.x)
  const y = Math.max(bounds.y, area.y)
  const right = Math.min(bounds.x + bounds.width, area.x + area.width)
  const bottom = Math.min(bounds.y + bounds.height, area.y + area.height)

  return right > x && bottom > y ? { height: bottom - y, width: right - x, x, y } : null
}

const subtract = (spans: Span[], cut: Span): Span[] =>
  spans.flatMap(span => {
    if (cut.right <= span.left || cut.left >= span.right) {
      return [span]
    }

    const pieces: Span[] = []

    if (cut.left > span.left) {
      pieces.push({ left: span.left, right: Math.min(span.right, cut.left) })
    }

    if (cut.right < span.right) {
      pieces.push({ left: Math.max(span.left, cut.right), right: span.right })
    }

    return pieces
  })

/**
 * Convert front-to-back app-window rectangles into visible horizontal ledges.
 * A front window subtracts the part of every lower window edge it covers, so
 * the pet never appears to stand on an edge hidden behind another app. The
 * work-area bottom is always the first/fallback ledge.
 */
export function overlayRoamLedges(
  { visualLedges = [], windows, workArea }: OverlayRoamEnvironment,
  overlayWidth: number,
  petW: number,
  petH: number
): Ledge[] {
  const spriteOffsetX = Math.max(0, (overlayWidth - petW) / 2)

  const toLedge = (span: Span, y: number): Ledge => ({
    left: span.left - spriteOffsetX,
    right: span.right - spriteOffsetX - petW,
    y
  })

  const ledges: Ledge[] = [
    toLedge({ left: workArea.x, right: workArea.x + workArea.width }, workArea.y + workArea.height)
  ]

  const occluders: OverlayBounds[] = []

  for (const windowBounds of windows) {
    const clipped = clipTo(windowBounds, workArea)

    if (!clipped) {
      continue
    }

    const surfaceY = windowBounds.y
    let visible: Span[] = [{ left: clipped.x, right: clipped.x + clipped.width }]

    for (const front of occluders) {
      if (front.y <= surfaceY && front.y + front.height > surfaceY) {
        visible = subtract(visible, { left: front.x, right: front.x + front.width })
      }
    }

    // A window flush with the display top has no room for the sprite above it;
    // a taskbar/shell surface at the work-area bottom duplicates the floor.
    if (surfaceY - petH >= workArea.y && surfaceY < workArea.y + workArea.height - 8) {
      for (const span of visible) {
        const ledge = toLedge(span, surfaceY)

        if (ledge.right > ledge.left + 2) {
          ledges.push(ledge)
        }
      }
    }

    occluders.push(clipped)
  }

  for (const surface of visualLedges) {
    const left = Math.max(surface.left, workArea.x)
    const right = Math.min(surface.right, workArea.x + workArea.width)

    if (surface.y - petH < workArea.y || surface.y >= workArea.y + workArea.height - 8) {
      continue
    }

    const ledge = toLedge({ left, right }, surface.y)

    const duplicatesNative = ledges.some(
      existing =>
        Math.abs(existing.y - ledge.y) <= LEDGE_MATCH_TOLERANCE_PX &&
        Math.min(existing.right, ledge.right) > Math.max(existing.left, ledge.left) + 2
    )

    if (!duplicatesNative && ledge.right > ledge.left + 2) {
      ledges.push(ledge)
    }
  }

  return ledges
}

/** Top coordinate where the overlay's visible feet meet the desktop floor. */
export function overlayGroundY(workArea: OverlayBounds, windowHeight: number): number {
  return groundTop(
    { left: workArea.x, right: workArea.x + workArea.width, y: workArea.y + workArea.height },
    Math.max(1, windowHeight - PET_FOOT_INSET_PX)
  )
}

/** Whether the overlay is already resting close enough to a detected surface. */
export function overlaySupportAt(
  ledges: Ledge[],
  bounds: OverlayBounds,
  tolerancePx = SUPPORT_SNAP_TOLERANCE_PX
): Ledge | null {
  const footHeight = Math.max(1, bounds.height - PET_FOOT_INSET_PX)

  return (
    ledges.find(
      ledge =>
        bounds.x >= ledge.left - 2 &&
        bounds.x <= ledge.right + 2 &&
        Math.abs(bounds.y - groundTop(ledge, footHeight)) <= tolerancePx
    ) ?? null
  )
}

export function overlayHasSupport(ledges: Ledge[], bounds: OverlayBounds): boolean {
  return overlaySupportAt(ledges, bounds) !== null
}

const sameOverlayLedge = (a: Ledge, b: Ledge): boolean =>
  Math.abs(a.y - b.y) <= LEDGE_MATCH_TOLERANCE_PX && Math.min(a.right, b.right) > Math.max(a.left, b.left) + 2

const appendOverlayLedge = (ledges: Ledge[], ledge: Ledge): Ledge[] =>
  ledges.some(existing => sameOverlayLedge(existing, ledge)) ? ledges : [...ledges, ledge]

export function revalidateOverlayPlannedHop(
  planned: Ledge | null,
  freshLedges: Ledge[],
  previousFailures: number
): { failures: number; ledge: Ledge | null } {
  if (!planned) {
    return { failures: 0, ledge: null }
  }

  const live = freshLedges.find(ledge => sameOverlayLedge(ledge, planned))

  if (live) {
    return { failures: 0, ledge: live }
  }

  const failures = Math.max(0, Math.floor(previousFailures)) + 1

  return {
    failures,
    ledge: failures >= PLANNED_HOP_FAILURE_LIMIT ? null : planned
  }
}

/** Keep the pre-validated hop destination eligible while newly seen visual edges stabilize. */
export function overlayMotionLandingLedges(
  native: Ledge[],
  confirmedVisual: Ledge | null,
  plannedHop: Ledge | null
): Ledge[] {
  const detected = confirmedVisual ? appendOverlayLedge(native, confirmedVisual) : native

  return plannedHop ? appendOverlayLedge(detected, plannedHop) : detected
}

/** A chosen destination is authoritative when its planned hop reaches the endpoint. */
export function overlayHopEndpointLanding(
  plannedHop: Ledge | null,
  elapsedMs: number,
  durationMs: number
): Ledge | null {
  return plannedHop && elapsedMs >= durationMs ? plannedHop : null
}

/** Ignore an asynchronous surface probe once its movement phase has changed. */
export const overlayMotionProbeIsCurrent = (
  requestedEpoch: number,
  currentEpoch: number,
  requestedPhase: Phase,
  currentPhase: Phase
): boolean => requestedEpoch === currentEpoch && requestedPhase === currentPhase

export function advanceOverlayVisualCandidate(
  previous: OverlayVisualCandidate | null,
  visual: Ledge | null
): OverlayVisualCandidate | null {
  if (!visual) {
    return null
  }

  return previous && sameOverlayLedge(previous.ledge, visual)
    ? { hits: previous.hits + 1, ledge: visual }
    : { hits: 1, ledge: visual }
}

/** Relative surface height: 0 at the work-area top and 1 at its bottom. */
export function overlayLowerPosition(surfaceY: number, workArea: OverlayBounds): number {
  const height = Math.max(1, workArea.height)

  return Math.max(0, Math.min(1, (surfaceY - workArea.y) / height))
}

/** Keep walking at 50%; shift the vertical half from drops to hops near the bottom. */
export function overlayIdleAction(
  canDrop: boolean,
  rng: () => number = Math.random,
  lowerPosition = 0.4
): OverlayIdleAction {
  const sample = Math.max(0, Math.min(1, rng()))
  const heightBias = Math.max(0, Math.min(1, lowerPosition))

  const hopChance = canDrop
    ? MIN_HOP_CHANCE + (VERTICAL_ACTION_CHANCE - MIN_HOP_CHANCE) * heightBias
    : VERTICAL_ACTION_CHANCE

  if (sample < hopChance) {
    return 'hop'
  }

  return canDrop && sample < VERTICAL_ACTION_CHANCE ? 'drop' : 'walk'
}

/** Never intentionally drop without a pre-scanned landing destination below. */
export const overlayDropAllowed = (isElevated: boolean, hasDropDestination: boolean): boolean =>
  isElevated && hasDropDestination

export function overlayPlannedAction(
  canDrop: boolean,
  supportMisses: number,
  forceWalk: boolean,
  rng: () => number = Math.random,
  lowerPosition = 0.4
): OverlayIdleAction {
  return supportMisses > 0 || forceWalk ? 'walk' : overlayIdleAction(canDrop, rng, lowerPosition)
}

export function overlaySupportMissOutcome(previousFailures: number): {
  failures: number
  shouldDrop: boolean
} {
  const failures = Math.max(0, Math.floor(previousFailures)) + 1

  return { failures, shouldDrop: failures >= SUPPORT_FAILURE_LIMIT }
}

/** Treat small capture/rounding jitter around a resting height as grounded. */
export function overlayVerticalCorrection(currentY: number, restingY: number): OverlayVerticalCorrection {
  if (currentY < restingY - SUPPORT_SNAP_TOLERANCE_PX) {
    return 'fall'
  }

  if (currentY > restingY + SUPPORT_SNAP_TOLERANCE_PX) {
    return 'hop'
  }

  return 'snap'
}

/** First valid surface crossed while a falling or descending pet moves. */
export function overlayLandingAlongPath(
  ledges: Ledge[],
  from: Pick<OverlayBounds, 'x' | 'y'>,
  to: Pick<OverlayBounds, 'x' | 'y'>,
  windowHeight: number
): Ledge | null {
  if (to.y <= from.y) {
    return null
  }

  const footHeight = Math.max(1, windowHeight - PET_FOOT_INSET_PX)
  const travelY = to.y - from.y
  let landing: Ledge | null = null
  let landingY = Number.POSITIVE_INFINITY

  for (const ledge of ledges) {
    const restingY = groundTop(ledge, footHeight)

    if (restingY < from.y - 1 || restingY > to.y + 1 || restingY >= landingY) {
      continue
    }

    const progress = Math.max(0, Math.min(1, (restingY - from.y) / travelY))
    const crossingX = from.x + (to.x - from.x) * progress

    if (crossingX >= ledge.left - 2 && crossingX <= ledge.right + 2) {
      landing = ledge
      landingY = restingY
    }
  }

  return landing
}

/** Keep the visible sprite on-screen while allowing transparent window padding off-screen. */
export function clampOverlayRoamBounds(
  workArea: OverlayBounds,
  bounds: OverlayBounds,
  petW: number = bounds.width,
  petH: number = bounds.height
): OverlayBounds {
  const spriteOffsetX = Math.max(0, (bounds.width - petW) / 2)
  const minX = workArea.x - spriteOffsetX
  const maxX = Math.max(minX, workArea.x + workArea.width - spriteOffsetX - petW)
  const spriteOffsetY = Math.max(0, bounds.height - PET_FOOT_INSET_PX - petH)
  const maxY = overlayGroundY(workArea, bounds.height)
  const minY = Math.min(maxY, workArea.y - spriteOffsetY)

  return {
    ...bounds,
    x: Math.max(minX, Math.min(maxX, bounds.x)),
    y: Math.max(minY, Math.min(maxY, bounds.y))
  }
}

/**
 * Wander the entire transparent BrowserWindow across the desktop. The usable
 * display bottom and visible top edges of other apps are platformer surfaces:
 * the pet walks along them, falls when a surface disappears, and hops between
 * overlapping ledges. Enumeration failure degrades to the display floor.
 */
export function startPetOverlayRoam({
  enabled,
  isInteracting,
  loopMs,
  petH,
  petW,
  replanKey = 0
}: OverlayRoamOptions): () => void {
  const api = window.hermesDesktop?.petOverlay

  if (!enabled || !api?.roamEnvironment) {
    $petMotion.set(null)
    $petRoamDir.set(0)

    return () => {}
  }

  let stopped = false
  let raf = 0
  let timer = 0
  let lastFrame = 0
  let lastPaintX = Number.NaN
  let lastPaintY = Number.NaN
  let current: OverlayBounds | null = null
  let ledges: Ledge[] = []
  let motionLandingLedges: Ledge[] = []
  let plannedHopLanding: Ledge | null = null
  let currentLedge: Ledge | null = null
  let targetLedge: Ledge | null = null
  let phase: Phase = 'walk'
  let targetX = 0
  let startX = 0
  let startY = 0
  let phaseStarted = 0
  let fallVelocity = 0
  let hopApexY = 0
  let hopDuration = BASE_HOP_DURATION_MS
  let rememberedSupport: Ledge | null = null
  let supportMisses = 0
  let nextSupportRetryAt = 0
  let ignoredDropSurfaceY: number | null = null
  let settleFirstPlan = replanKey > 0
  let forceWalkNext = replanKey > 0
  const speed = (petW * 0.8) / Math.max(0.25, loopMs / 1000)

  const footHeight = (): number => Math.max(1, (current?.height ?? 1) - PET_FOOT_INSET_PX)
  const restY = (ledge: Ledge): number => groundTop(ledge, footHeight())

  const signal = (pose: 'jump' | 'run' | null, dir: -1 | 0 | 1 = 0) => {
    $petMotion.set(pose)
    $petRoamDir.set(pose === 'run' ? dir : 0)
  }

  const remember = () => {
    if (current) {
      api.control({ bounds: current, type: 'bounds' })
    }
  }

  const schedulePlan = (delayMs: number = dwellMs(OVERLAY_PAUSE_DWELL)) => {
    signal(null)
    remember()
    timer = window.setTimeout(plan, delayMs)
  }

  const syncFromWindow = () => {
    current = {
      height: window.outerHeight,
      width: window.outerWidth,
      x: window.screenX,
      y: window.screenY
    }
  }

  const paint = (force = false) => {
    if (current) {
      const x = Math.round(current.x)
      const y = Math.round(current.y)

      if (!force && x === lastPaintX && y === lastPaintY) {
        return
      }

      lastPaintX = x
      lastPaintY = y
      api.setBounds(current)
    }
  }

  const beginHop = (ledge: Ledge, x: number, now: number) => {
    if (!current) {
      return
    }

    rememberedSupport = null
    supportMisses = 0
    nextSupportRetryAt = 0
    ignoredDropSurfaceY = null
    plannedHopLanding = ledge
    motionLandingLedges = appendOverlayLedge(ledges, plannedHopLanding)
    forceWalkNext = true
    targetLedge = ledge
    targetX = x
    startX = current.x
    startY = current.y
    phase = 'hop'
    phaseStarted = now
    const deltaY = restY(ledge) - startY
    const travel = Math.hypot(targetX - startX, deltaY)

    hopApexY = overlayHopApexY(petH, restY(ledge))
    hopDuration = Math.min(1600, BASE_HOP_DURATION_MS + travel * 0.7)
    signal('jump')
  }

  const beginDrop = (surface: Ledge, destination: Ledge) => {
    if (!current) {
      return
    }

    rememberedSupport = null
    supportMisses = 0
    nextSupportRetryAt = 0
    ignoredDropSurfaceY = surface.y
    plannedHopLanding = null
    motionLandingLedges = [destination]
    forceWalkNext = true
    targetLedge = destination
    phase = 'fall'
    fallVelocity = 0
    signal('jump')
  }

  const beginMotion = (now: number, workArea: OverlayBounds, settleAfterDrag = false): boolean => {
    if (!current || ledges.length === 0) {
      return false
    }

    currentLedge = resolveLedge(ledges, current.x, current.y, footHeight())
    const restingY = restY(currentLedge)

    if (currentLedge !== ledges[0]) {
      rememberedSupport = currentLedge
    }

    const verticalCorrection = overlayVerticalCorrection(current.y, restingY)

    if (verticalCorrection === 'fall') {
      targetLedge = currentLedge
      phase = 'fall'
      fallVelocity = 0
      plannedHopLanding = null
      motionLandingLedges = ledges
      forceWalkNext = true
      signal('jump')

      return true
    }

    if (verticalCorrection === 'hop') {
      beginHop(currentLedge, current.x, now)

      return true
    }

    current.y = restingY
    const fromLedge = currentLedge
    const fromX = current.x
    const reachable = overlayHopDestinations(ledges, fromLedge, fromX, petH)
    const dropDestinations = overlayDropDestinations(ledges, fromLedge, fromX)
    const lowerPosition = overlayLowerPosition(fromLedge.y, workArea)

    const idleAction = overlayPlannedAction(
      overlayDropAllowed(currentLedge !== ledges[0], dropDestinations.length > 0),
      supportMisses,
      settleAfterDrag || forceWalkNext,
      Math.random,
      lowerPosition
    )

    if (idleAction === 'drop') {
      const destination = randomOverlayDropDestination(dropDestinations)

      if (destination) {
        beginDrop(currentLedge, destination.ledge)

        return true
      }
    }

    if (idleAction === 'hop') {
      const next = nearestOverlayHopDestination(reachable, fromLedge.y)

      if (next) {
        beginHop(next.ledge, fromX, now)

        return true
      }
    }

    targetX = pickStrollTarget(currentLedge, current.x)

    if (Math.abs(targetX - current.x) < Math.min(MIN_TRAVEL_PX, (currentLedge.right - currentLedge.left) / 2)) {
      targetX = current.x < (currentLedge.left + currentLedge.right) / 2 ? currentLedge.right : currentLedge.left
    }

    const dir = targetX > current.x ? 1 : targetX < current.x ? -1 : 0

    if (dir === 0) {
      return false
    }

    forceWalkNext = false
    plannedHopLanding = null
    targetLedge = currentLedge
    phase = 'walk'
    signal('run', dir)

    return true
  }

  const environmentLedges = (environment: OverlayRoamEnvironment) =>
    overlayRoamLedges(environment, current?.width ?? 1, petW, petH)

  const updateMotionLandingLedges = (freshLedges: Ledge[]) => {
    motionLandingLedges = plannedHopLanding ? appendOverlayLedge(freshLedges, plannedHopLanding) : freshLedges
  }

  const reconcileRememberedSupport = (
    freshLedges: Ledge[],
    now: number
  ): { ledges: Ledge[]; lostSupport: Ledge | null } => {
    if (!current || !rememberedSupport || !overlayHasSupport([rememberedSupport], current)) {
      rememberedSupport = null
      supportMisses = 0
      nextSupportRetryAt = 0

      return { ledges: freshLedges, lostSupport: null }
    }

    const detectedSupport = overlaySupportAt(freshLedges, current)

    if (detectedSupport) {
      rememberedSupport = detectedSupport
      supportMisses = 0
      nextSupportRetryAt = 0

      return { ledges: freshLedges, lostSupport: null }
    }

    if (supportMisses === 0 || now >= nextSupportRetryAt) {
      const miss = overlaySupportMissOutcome(supportMisses)

      supportMisses = miss.failures
      nextSupportRetryAt = now + SUPPORT_RETRY_MS

      if (miss.shouldDrop) {
        const lostSupport = rememberedSupport

        rememberedSupport = null
        supportMisses = 0
        nextSupportRetryAt = 0

        return { ledges: freshLedges, lostSupport }
      }
    }

    return { ledges: appendOverlayLedge(freshLedges, rememberedSupport), lostSupport: null }
  }

  const step = (now: number) => {
    raf = 0

    if (stopped || !current || !targetLedge) {
      return
    }

    if (isInteracting()) {
      syncFromWindow()
      plannedHopLanding = null
      signal(null)
      timer = window.setTimeout(plan, PAUSE_POLL_MS)

      return
    }

    const dt = Math.min(MAX_DT_S, Math.max(0, now - lastFrame) / 1000)
    const targetY = restY(targetLedge)
    const previous = { x: current.x, y: current.y }
    lastFrame = now

    if (phase === 'fall') {
      fallVelocity += GRAVITY_PX_S2 * dt
      current.y = Math.min(restY(ledges[0]!), current.y + fallVelocity * dt)
    } else if (phase === 'hop') {
      const progress = Math.min(1, (now - phaseStarted) / hopDuration)
      current.x = startX + (targetX - startX) * progress
      current.y = overlayHopYAtProgress(startY, hopApexY, targetY, progress)
    } else {
      const remaining = targetX - current.x
      const distance = speed * dt
      current.x = Math.abs(remaining) <= distance ? targetX : current.x + Math.sign(remaining) * distance
      current.y = targetY
    }

    const liveLandingLedges = motionLandingLedges.length > 0 ? motionLandingLedges : [ledges[0]!]

    const landingLedges =
      ignoredDropSurfaceY === null
        ? liveLandingLedges
        : liveLandingLedges.filter(ledge => Math.abs(ledge.y - ignoredDropSurfaceY!) > LEDGE_MATCH_TOLERANCE_PX)

    const pathLanding =
      phase === 'walk'
        ? null
        : (overlayLandingAlongPath(landingLedges, previous, current, current.height) ??
          overlayHopEndpointLanding(plannedHopLanding, now - phaseStarted, hopDuration))

    if (pathLanding) {
      targetLedge = pathLanding
      current.y = restY(pathLanding)
      plannedHopLanding = null
      rememberedSupport = pathLanding === liveLandingLedges[0] ? null : pathLanding
      supportMisses = 0
      ignoredDropSurfaceY = null
    }

    let continuingFall = false

    if (!pathLanding && phase === 'hop' && now - phaseStarted >= hopDuration) {
      // The planned landing disappeared or was never crossed. Preserve the
      // final horizontal position and continue downward until a live surface
      // on the path is actually crossed.
      phase = 'fall'
      plannedHopLanding = null
      targetLedge = resolveLedge(ledges, current.x, current.y, footHeight())
      fallVelocity = 0
      continuingFall = true
    }

    const arrived = pathLanding !== null || (!continuingFall && phase === 'walk' && current.x === targetX)

    if (arrived) {
      if (!pathLanding) {
        current.x = phase === 'fall' ? current.x : targetX
        current.y = targetY
      }

      paint(true)

      const retryDelay = supportMisses > 0 ? Math.max(0, nextSupportRetryAt - performance.now()) : undefined

      // Stop on the landing first. The next plan captures the now-stationary
      // scene and traces the support sideways before choosing a walk target.
      schedulePlan(pathLanding ? LANDING_ANALYSIS_DELAY_MS : retryDelay)
    } else {
      paint()
      raf = window.requestAnimationFrame(step)
    }
  }

  const plan = async () => {
    timer = 0

    if (stopped) {
      return
    }

    if (isInteracting()) {
      syncFromWindow()
      timer = window.setTimeout(plan, PAUSE_POLL_MS)

      return
    }

    const environment = await api.roamEnvironment({
      petHeight: petH,
      petWidth: petW,
      scanMode: 'destination'
    })

    if (stopped) {
      return
    }

    if (!environment) {
      timer = window.setTimeout(plan, 1000)

      return
    }

    const raw = {
      height: window.outerHeight,
      width: window.outerWidth,
      x: window.screenX,
      y: window.screenY
    }

    current = clampOverlayRoamBounds(environment.workArea, raw, petW, petH)
    const now = performance.now()
    const freshLedges = environmentLedges(environment)

    updateMotionLandingLedges(freshLedges)

    const support = reconcileRememberedSupport(freshLedges, now)

    ledges = support.ledges

    if (supportMisses > 0 && !support.lostSupport) {
      // Keep the pet still while retrying a missing visual support. Capturing
      // and contrast analysis during movement makes native window motion hitch.
      schedulePlan(Math.max(0, nextSupportRetryAt - now))

      return
    }

    const settleAfterDrag = settleFirstPlan

    settleFirstPlan = false

    const draggedSupport = settleAfterDrag ? overlaySupportAt(ledges, current, DRAG_SETTLE_TOLERANCE_PX) : null

    if (draggedSupport) {
      current.y = restY(draggedSupport)
      rememberedSupport = draggedSupport
      supportMisses = 0
      nextSupportRetryAt = 0
    }

    if (support.lostSupport) {
      // The third miss starts a straight fall. A flicker of the same old line
      // cannot immediately catch the pet again, but lower path surfaces can.
      ignoredDropSurfaceY = support.lostSupport.y
    }

    if (!beginMotion(now, environment.workArea, Boolean(draggedSupport))) {
      schedulePlan()

      return
    }

    lastFrame = now
    raf = window.requestAnimationFrame(step)
  }

  void plan()

  return () => {
    stopped = true
    window.clearTimeout(timer)
    window.cancelAnimationFrame(raf)
    signal(null)
  }
}
