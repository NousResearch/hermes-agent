import { atom } from 'nanostores'

import { persistBoolean, persistString, storedBoolean, storedString } from '@/lib/storage'
import { $petActivity, $petInfo, $petUnread, clearPetUnread, type PetActivity, type PetInfo } from '@/store/pet'
import { $awaitingResponse, $busy } from '@/store/session'

/**
 * Controller for the pop-out pet overlay (main-renderer side).
 *
 * Shift-clicking the in-window pet "pops it out" into a transparent,
 * always-on-top OS window (created in electron/main.ts) that can leave the
 * app's bounds and stays visible while Hermes is minimized. That window carries
 * NO gateway connection — this renderer remains the single source of truth and
 * pushes the live pet state to it over IPC. Control flows back (pop the pet back
 * in, submit a composer message) via `onControl`.
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
  info: PetInfo
  activity: PetActivity
  busy: boolean
  awaiting: boolean
  /** Drives the overlay's mail icon: a finish landed while you were away. */
  unread: boolean
}

export type PetOverlayControl =
  | { type: 'pop-in' }
  | { type: 'ready' }
  | { type: 'submit'; text: string }
  | { type: 'bounds'; bounds: PetOverlayBounds }
  | { type: 'open-app' }
  | { type: 'toggle-app' }
  | { type: 'scale'; scale: number }

// Persisted across restarts: was the pet popped out, and where on the desktop
// did the user leave it. Keyed v1; bump if the bounds shape ever changes.
const OVERLAY_ACTIVE_KEY = 'hermes.desktop.pet-overlay-active.v1'
const OVERLAY_BOUNDS_KEY = 'hermes.desktop.pet-overlay-bounds.v1'

export const $petOverlayActive = atom(storedBoolean(OVERLAY_ACTIVE_KEY, false))

// Persist the in/out choice so a popped-out pet comes back popped out.
$petOverlayActive.subscribe(active => persistBoolean(OVERLAY_ACTIVE_KEY, active))

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

/**
 * Window bounds (width/height) that fully contain the pet at a given scale, plus
 * the padding for its bubble/composer/drag margins. The single source of truth
 * for both the initial pop-out size and the live wheel-to-scale resize, so the
 * sprite is never cropped by the window edge no matter how big it's scaled.
 */
export function overlayWindowSize(frameW: number, frameH: number, scale: number): { width: number; height: number } {
  return {
    width: Math.max(OVERLAY_MIN_W, Math.round(frameW * scale + OVERLAY_PAD_X)),
    height: Math.max(OVERLAY_MIN_H, Math.round(frameH * scale + OVERLAY_PAD_Y))
  }
}

let stateUnsubs: Array<() => void> = []
let controlUnsub: (() => void) | null = null
let submitHandler: ((text: string) => void) | null = null
let openAppHandler: (() => void) | null = null
let scaleHandler: ((scale: number) => void) | null = null

function currentPayload(): PetOverlayStatePayload {
  return {
    info: $petInfo.get(),
    activity: $petActivity.get(),
    busy: $busy.get(),
    awaiting: $awaitingResponse.get(),
    unread: $petUnread.get()
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
    $petUnread.subscribe(pushNow)
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

/**
 * Wire the overlay→renderer control channel once. Returns a disposer. Idempotent
 * — a second call while already wired is a no-op.
 */
export function initPetOverlayBridge(): () => void {
  const api = window.hermesDesktop?.petOverlay

  if (!api || controlUnsub) {
    return () => {}
  }

  controlUnsub = api.onControl(payload => {
    if (payload?.type === 'pop-in') {
      popInPet()
    } else if (payload?.type === 'ready') {
      // The overlay just mounted — hand it the current frame.
      pushNow()
    } else if (payload?.type === 'submit' && typeof payload.text === 'string') {
      submitHandler?.(payload.text)
    } else if (payload?.type === 'bounds' && payload.bounds) {
      // The user dragged the overlay to a new desktop spot — remember it.
      saveBounds(payload.bounds)
    } else if (payload?.type === 'scale' && typeof payload.scale === 'number') {
      // The user resized the popped-out pet (Alt+wheel) — persist it through
      // the main renderer's gateway; the new scale rides $petInfo back to the
      // overlay on the next push, keeping both surfaces in sync.
      scaleHandler?.(payload.scale)
    } else if (payload?.type === 'open-app') {
      // Mail icon: surface the app on the most recent thread (main.ts already
      // focused the window before forwarding this) and mark it read.
      clearPetUnread()
      openAppHandler?.()
    }
  })

  return () => {
    controlUnsub?.()
    controlUnsub = null
  }
}
