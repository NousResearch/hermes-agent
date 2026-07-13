import { atom } from 'nanostores'

import { persistBoolean, persistString, storedBoolean, storedString } from '@/lib/storage'
import { $petActivity, $petInfo, $petUnread, clearPetUnread, type PetActivity, type PetInfo } from '@/store/pet'
import { $petActionCenter, type PetActionCenterState } from '@/store/pet-action-center'
import { $awaitingResponse, $busy } from '@/store/session'

/**
 * Controller for the pop-out pet overlay (main-renderer side).
 *
 * Shift-clicking the in-window pet "pops it out" into a transparent,
 * always-on-top OS window (created in electron/main.ts) that can leave the
 * app's bounds and stays visible while Hermes is minimized. That window carries
 * NO gateway connection — this renderer remains the single source of truth and
 * pushes the live pet state to it over IPC. Narrow typed controls flow back
 * (window gestures, composer submit, action-center intents) via `onControl`.
 *
 * The overlay renders the same `PetSprite` / `PetBubble` as the in-window pet by
 * mirroring the four reactive inputs of `$petState` (`$petInfo`, `$petActivity`,
 * `$busy`, `$awaitingResponse`) into its own copies of those atoms — so the
 * popped-out mascot is pixel-identical and needs zero bespoke render logic.
 */

export interface PetOverlayBounds {
  x: number
  y: number
  width: number
  height: number
}

/**
 * Request to open the overlay window. `screen` says whether `bounds` are already
 * in absolute screen coordinates (a remembered/dragged spot) or in the main
 * window's viewport space (a fresh shift-click pop-out, which main.ts converts
 * by adding the content origin).
 */
export interface PetOverlayOpenRequest {
  bounds: PetOverlayBounds
  screen?: boolean
}

/** Everything the overlay needs to reproduce the live mascot. */
export interface PetOverlayStatePayload {
  actionCenter: PetActionCenterState
  info: PetInfo
  activity: PetActivity
  busy: boolean
  awaiting: boolean
  /** Drives the overlay's mail icon: a finish landed while you were away. */
  unread: boolean
  /** Latest reaction — bumping its id forwards a burst to the overlay. */
  reaction: PetReaction | null
}

export type PetActionCenterApprovalChoice = 'approve-always' | 'approve-once' | 'approve-session' | 'deny'

export type PetActionCenterControl =
  | { type: 'action-center-select'; itemId: string }
  | {
      type: 'action-center-approval'
      itemId: string
      choice: PetActionCenterApprovalChoice
      reason?: string
    }
  | { type: 'action-center-clarify'; itemId: string; answer: string }
  | { type: 'action-center-open-session'; itemId: string }
  | { type: 'action-center-submit'; itemId: string; text: string }
  | { type: 'action-center-steer'; itemId: string; text: string }
  | { type: 'action-center-queue'; itemId: string; text: string }
  | { type: 'action-center-stop'; itemId: string }
  | { type: 'action-center-acknowledge'; itemId: string }

export type PetOverlayControl =
  | { type: 'pop-in' }
  | { type: 'ready' }
  | { type: 'submit'; text: string }
  | { type: 'bounds'; bounds: PetOverlayBounds }
  | { type: 'open-app' }
  | { type: 'toggle-app' }
  | { type: 'scale'; scale: number }
  | PetActionCenterControl

const APPROVAL_CHOICES = new Set<PetActionCenterApprovalChoice>([
  'approve-always',
  'approve-once',
  'approve-session',
  'deny'
])

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function controlItemId(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

function hasExactOwnKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value)

  return actual.length === keys.length && keys.every(key => Object.prototype.hasOwnProperty.call(value, key))
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isApprovalChoice(value: unknown): value is PetActionCenterApprovalChoice {
  return typeof value === 'string' && APPROVAL_CHOICES.has(value as PetActionCenterApprovalChoice)
}

/**
 * Treat IPC as an untrusted runtime boundary. Rebuild accepted controls from
 * their narrow fields so injected profile/session/route values never reach the
 * main-renderer action handler, even when a malformed overlay sends extras.
 */
export function parsePetOverlayControl(value: unknown): PetOverlayControl | null {
  if (!isRecord(value) || typeof value.type !== 'string') {
    return null
  }

  if (value.type === 'pop-in' || value.type === 'ready' || value.type === 'open-app' || value.type === 'toggle-app') {
    return { type: value.type }
  }

  switch (value.type) {
    case 'submit':
      return typeof value.text === 'string' ? { type: value.type, text: value.text } : null
    case 'bounds': {
      if (!isRecord(value.bounds)) {
        return null
      }

      const { height, width, x, y } = value.bounds

      return isFiniteNumber(height) && isFiniteNumber(width) && isFiniteNumber(x) && isFiniteNumber(y)
        ? { type: value.type, bounds: { height, width, x, y } }
        : null
    }

    case 'scale':
      return typeof value.scale === 'number' && Number.isFinite(value.scale)
        ? { type: value.type, scale: value.scale }
        : null

    case 'action-center-select':

    case 'action-center-open-session':

    case 'action-center-stop':
    case 'action-center-acknowledge': {
      const itemId = controlItemId(value.itemId)

      return itemId && hasExactOwnKeys(value, ['type', 'itemId']) ? { type: value.type, itemId } : null
    }

    case 'action-center-clarify': {
      const itemId = controlItemId(value.itemId)

      return itemId && typeof value.answer === 'string' && hasExactOwnKeys(value, ['type', 'itemId', 'answer'])
        ? { type: value.type, itemId, answer: value.answer }
        : null
    }

    case 'action-center-submit':

    case 'action-center-steer':
    case 'action-center-queue': {
      const itemId = controlItemId(value.itemId)

      return itemId && typeof value.text === 'string' && hasExactOwnKeys(value, ['type', 'itemId', 'text'])
        ? { type: value.type, itemId, text: value.text }
        : null
    }

    case 'action-center-approval': {
      const itemId = controlItemId(value.itemId)
      const choice = value.choice

      if (!itemId || !isApprovalChoice(choice)) {
        return null
      }

      if (
        !hasExactOwnKeys(
          value,
          value.reason === undefined ? ['type', 'itemId', 'choice'] : ['type', 'itemId', 'choice', 'reason']
        )
      ) {
        return null
      }

      if (choice !== 'deny' && value.reason !== undefined) {
        return null
      }

      if (value.reason !== undefined && typeof value.reason !== 'string') {
        return null
      }

      return value.reason === undefined
        ? { type: value.type, itemId, choice }
        : { type: value.type, itemId, choice, reason: value.reason }
    }

    default:
      return null
  }
}

function isPetActionCenterControl(control: PetOverlayControl): control is PetActionCenterControl {
  return (
    control.type === 'action-center-select' ||
    control.type === 'action-center-approval' ||
    control.type === 'action-center-clarify' ||
    control.type === 'action-center-open-session' ||
    control.type === 'action-center-submit' ||
    control.type === 'action-center-steer' ||
    control.type === 'action-center-queue' ||
    control.type === 'action-center-stop' ||
    control.type === 'action-center-acknowledge'
  )
}

// Persisted across restarts: was the pet popped out, and where on the desktop
// did the user leave it. Keyed v1; bump if the bounds shape ever changes.
const OVERLAY_ACTIVE_KEY = 'hermes.desktop.pet-overlay-active.v1'
const OVERLAY_BOUNDS_KEY = 'hermes.desktop.pet-overlay-bounds.v1'

export const $petOverlayActive = atom(storedBoolean(OVERLAY_ACTIVE_KEY, false))

// Persist the in/out choice so a popped-out pet comes back popped out.
$petOverlayActive.subscribe(active => persistBoolean(OVERLAY_ACTIVE_KEY, active))

/**
 * Reaction signal forwarded to the popped-out overlay window via the state
 * mirror below. `id` is a monotonic nonce so the overlay fires once per bump;
 * `kind` selects the renderer (today only `vibe` → hearts). Generic on purpose
 * so future reactions (emoji, etc.) ride the same channel.
 */
export interface PetReaction {
  id: number
  kind: string
}

export const $petReaction = atom<PetReaction | null>(null)

export const forwardPetReaction = (kind: string) => $petReaction.set({ id: ($petReaction.get()?.id ?? 0) + 1, kind })

function loadSavedBounds(): null | PetOverlayBounds {
  try {
    const raw = storedString(OVERLAY_BOUNDS_KEY)

    if (!raw) {
      return null
    }

    const parsed = JSON.parse(raw) as Partial<PetOverlayBounds>

    if (
      typeof parsed.x === 'number' &&
      typeof parsed.y === 'number' &&
      typeof parsed.width === 'number' &&
      typeof parsed.height === 'number'
    ) {
      return { height: parsed.height, width: parsed.width, x: parsed.x, y: parsed.y }
    }
  } catch {
    // fall through to null
  }

  return null
}

function saveBounds(bounds: PetOverlayBounds): void {
  persistString(OVERLAY_BOUNDS_KEY, JSON.stringify(bounds))
}

// The overlay window is padded around the sprite so the bubble (above), the
// drag area, and the pop-up composer all have room; the pet sits near the
// bottom and the rest of the rectangle is transparent + click-through.
const OVERLAY_PAD_X = 100
const OVERLAY_PAD_Y = 200
const OVERLAY_MIN_W = 240
const OVERLAY_MIN_H = 300
const OVERLAY_CONTENT_ROOM_X = 32
const OVERLAY_CONTENT_ROOM_TOP = 16
const OVERLAY_CONTENT_ROOM_BOTTOM = 24

export interface PetOverlaySize {
  width: number
  height: number
}

export interface PetOverlayMeasuredContent {
  width: number
  height: number
}

export interface PetOverlayWheelAnchor {
  clientX: number
  clientY: number
  ratio: number
}

interface AnchoredOverlayBoundsInput {
  currentBounds: PetOverlayBounds
  targetSize: PetOverlaySize
  paddingBottom: number
  wheelAnchor?: PetOverlayWheelAnchor | null
}

const finiteNonNegative = (value: number): number => (Number.isFinite(value) && value > 0 ? value : 0)

/**
 * Window bounds (width/height) that fully contain the pet at a given scale, plus
 * the padding for its bubble/composer/drag margins. The single source of truth
 * for both the initial pop-out size and the live wheel-to-scale resize, so the
 * sprite is never cropped by the window edge no matter how big it's scaled.
 */
export function overlayWindowSize(frameW: number, frameH: number, scale: number): PetOverlaySize {
  const safeFrameW = finiteNonNegative(frameW)
  const safeFrameH = finiteNonNegative(frameH)
  const safeScale = finiteNonNegative(scale)

  return {
    width: Math.max(OVERLAY_MIN_W, Math.round(safeFrameW * safeScale + OVERLAY_PAD_X)),
    height: Math.max(OVERLAY_MIN_H, Math.round(safeFrameH * safeScale + OVERLAY_PAD_Y))
  }
}

/**
 * Window size for the live overlay. The compact pet geometry remains the floor;
 * an observed action-center/bubble/sprite stack can only grow it. Invalid DOM
 * measurements are treated as absent so a broken observer can never collapse
 * the native window below its safe compact dimensions.
 */
export function overlayWindowTargetSize(
  frameW: number,
  frameH: number,
  scale: number,
  measuredContent?: PetOverlayMeasuredContent | null
): PetOverlaySize {
  const compact = overlayWindowSize(frameW, frameH, scale)
  const measuredWidth = finiteNonNegative(measuredContent?.width ?? 0)
  const measuredHeight = finiteNonNegative(measuredContent?.height ?? 0)
  let width = Math.max(compact.width, Math.round(measuredWidth + OVERLAY_CONTENT_ROOM_X))

  // Match compact-width parity so integer screen bounds can preserve the
  // center exactly in both directions (odd/even width changes otherwise force
  // a half-pixel choice and accumulate a one-pixel expand/collapse drift).
  if ((width - compact.width) % 2 !== 0) {
    width += 1
  }

  return {
    width,
    height: Math.max(
      compact.height,
      Math.round(measuredHeight + OVERLAY_CONTENT_ROOM_TOP + OVERLAY_CONTENT_ROOM_BOTTOM)
    )
  }
}

/**
 * Resize about a stable screen-space anchor. Ordinary content/scale changes pin
 * the window's bottom-center (the pet's feet). Alt+wheel supplies the historical
 * cursor/ratio anchor formula verbatim so the pixel under the cursor stays put.
 */
export function anchoredOverlayBounds({
  currentBounds,
  targetSize,
  paddingBottom,
  wheelAnchor
}: AnchoredOverlayBoundsInput): PetOverlayBounds {
  if (!wheelAnchor) {
    return {
      height: targetSize.height,
      width: targetSize.width,
      x: Math.round(currentBounds.x + currentBounds.width / 2 - targetSize.width / 2),
      y: Math.round(currentBounds.y + currentBounds.height - targetSize.height)
    }
  }

  const { clientX: ax, clientY: ay, ratio } = wheelAnchor

  return {
    height: targetSize.height,
    width: targetSize.width,
    x: Math.round(currentBounds.x + ax - (ax - currentBounds.width / 2) * ratio - targetSize.width / 2),
    y: Math.round(
      currentBounds.y + ay - (ay - (currentBounds.height - paddingBottom)) * ratio - (targetSize.height - paddingBottom)
    )
  }
}

let stateUnsubs: Array<() => void> = []
let controlUnsub: (() => void) | null = null
let submitHandler: ((text: string) => void) | null = null
let openAppHandler: (() => void) | null = null
let scaleHandler: ((scale: number) => void) | null = null
let actionCenterHandler: ((control: PetActionCenterControl) => void) | null = null

function currentPayload(): PetOverlayStatePayload {
  return {
    actionCenter: $petActionCenter.get(),
    info: $petInfo.get(),
    activity: $petActivity.get(),
    busy: $busy.get(),
    awaiting: $awaitingResponse.get(),
    unread: $petUnread.get(),
    reaction: $petReaction.get()
  }
}

function pushNow(): void {
  window.hermesDesktop?.petOverlay?.pushState(currentPayload())
}

/**
 * Open the overlay window and start mirroring live state into it. The main
 * process echoes back the actual screen bounds it used, which we persist so the
 * pet reopens exactly where the user left it.
 */
function openOverlay(request: PetOverlayOpenRequest): void {
  const api = window.hermesDesktop?.petOverlay

  if (!api || stateUnsubs.length) {
    return
  }

  $petOverlayActive.set(true)
  void api.open(request).then(res => {
    if (res?.bounds) {
      saveBounds(res.bounds)
    }

    pushNow()
  })

  // Mirror live state into the overlay. subscribe() fires immediately, so the
  // overlay also gets a first frame the moment it's ready (it asks via 'ready').
  stateUnsubs = [
    $petInfo.subscribe(pushNow),
    $petActivity.subscribe(pushNow),
    $busy.subscribe(pushNow),
    $awaitingResponse.subscribe(pushNow),
    $petUnread.subscribe(pushNow),
    $petReaction.subscribe(pushNow),
    $petActionCenter.subscribe(pushNow)
  ]
}

/**
 * Pop the pet out of the window. `petRect` is the in-window sprite's viewport
 * rect; we grow it to the padded overlay size and center the window on the
 * pet's old spot (main.ts adds the window's screen origin). If the user has
 * popped out before, reopen at that remembered desktop spot instead.
 */
export function popOutPet(petRect: PetOverlayBounds): void {
  if ($petOverlayActive.get() || stateUnsubs.length) {
    return
  }

  const saved = loadSavedBounds()

  if (saved) {
    openOverlay({ bounds: saved, screen: true })

    return
  }

  // Size the window off the pet's scale (not the measured rect, which includes
  // the shadow) so it matches the live resize math exactly — no jump on open.
  const pet = $petInfo.get()
  const { width, height } = overlayWindowSize(pet.frameW ?? 192, pet.frameH ?? 208, pet.scale ?? 0.33)
  const x = Math.round(petRect.x - (width - petRect.width) / 2)
  const y = Math.round(petRect.y - (height - petRect.height) / 2)

  openOverlay({ bounds: { height, width, x, y }, screen: false })
}

/**
 * Restore the overlay on boot if the pet was popped out when the app last
 * closed. Requires a remembered desktop spot — without one we fall back to the
 * in-window pet rather than spawning an orphan window at the origin.
 */
export function restorePetOverlay(): void {
  if (!window.hermesDesktop?.petOverlay || !$petOverlayActive.get() || stateUnsubs.length) {
    return
  }

  const saved = loadSavedBounds()

  if (!saved) {
    $petOverlayActive.set(false)

    return
  }

  openOverlay({ bounds: saved, screen: true })
}

/** Pop the pet back into the window (closes the overlay window). */
export function popInPet(): void {
  for (const off of stateUnsubs) {
    off()
  }

  stateUnsubs = []
  $petOverlayActive.set(false)
  void window.hermesDesktop?.petOverlay?.close()
}

/** Register the handler that turns an overlay composer submit into a real send. */
export function setPetOverlaySubmitHandler(fn: ((text: string) => void) | null): void {
  submitHandler = fn
}

/** Register the handler that opens the app to the most recent thread (mail icon). */
export function setPetOverlayOpenAppHandler(fn: (() => void) | null): void {
  openAppHandler = fn
}

/** Register the handler that persists a scale resized via the overlay's Alt+wheel gesture. */
export function setPetOverlayScaleHandler(fn: ((scale: number) => void) | null): void {
  scaleHandler = fn
}

/** Register the main-renderer owner for typed action-center intents. */
export function setPetOverlayActionCenterHandler(fn: ((control: PetActionCenterControl) => void) | null): void {
  actionCenterHandler = fn
}

/**
 * Send a typed control from the overlay side. This is the gateway-less control
 * channel — the overlay calls this, the main renderer's bridge parses it via
 * `parsePetOverlayControl` and dispatches to registered handlers. No gateway
 * imports, no profile/session/route values are passed from the component.
 */
export function sendPetOverlayControl(control: PetOverlayControl): void {
  window.hermesDesktop?.petOverlay?.control(control)
}

/**
 * Wire the overlay→renderer control channel once. Returns a disposer. Idempotent
 * — a second call while already wired is a no-op.
 */
export function initPetOverlayBridge(): () => void {
  const api = window.hermesDesktop?.petOverlay

  if (!api || controlUnsub) {
    return () => {}
  }

  controlUnsub = api.onControl(rawPayload => {
    const payload = parsePetOverlayControl(rawPayload)

    if (!payload) {
      return
    }

    if (payload.type === 'pop-in') {
      popInPet()
    } else if (payload.type === 'ready') {
      // The overlay just mounted — hand it the current frame.
      pushNow()
    } else if (payload.type === 'submit' && typeof payload.text === 'string') {
      submitHandler?.(payload.text)
    } else if (payload.type === 'bounds' && payload.bounds) {
      // The user dragged the overlay to a new desktop spot — remember it.
      saveBounds(payload.bounds)
    } else if (payload.type === 'scale' && typeof payload.scale === 'number') {
      // The user resized the popped-out pet (Alt+wheel) — persist it through
      // the main renderer's gateway; the new scale rides $petInfo back to the
      // overlay on the next push, keeping both surfaces in sync.
      scaleHandler?.(payload.scale)
    } else if (payload.type === 'open-app') {
      // Mail icon: surface the app on the most recent thread (main.ts already
      // focused the window before forwarding this) and mark it read.
      clearPetUnread()
      openAppHandler?.()
    } else if (isPetActionCenterControl(payload)) {
      actionCenterHandler?.(payload)
    }
  })

  return () => {
    controlUnsub?.()
    controlUnsub = null
  }
}
