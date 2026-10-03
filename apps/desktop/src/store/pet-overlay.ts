import { atom } from 'nanostores'

import { PRIMARY_SESSION_VIEW } from '@/app/chat/session-view'
import { chatMessageText } from '@/lib/chat-messages/parts'
import { persistBoolean, persistString, storedBoolean, storedString } from '@/lib/storage'
import { $petActivity, $petInfo, $petUnread, clearPetUnread, type PetActivity, type PetInfo } from '@/store/pet'

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
 * mirroring the reactive inputs of `$petState` (`$petInfo`, `$petActivity`, the
 * primary view's turn-busy and awaiting-response) into its own copies of those atoms — so the
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
  /** Latest reaction — bumping its id forwards a burst to the overlay. */
  reaction: PetReaction | null
  /** Recent visible turns of the primary session, oldest first, so the
   *  companion card reads as a continuing conversation. */
  thread: PetOverlayTurn[]
  /** What dictation heard (or why it failed) — echoed into the balloon. */
  heard: PetOverlayHeard | null
}

export interface PetOverlayTurn {
  id: string
  role: 'user' | 'assistant'
  text: string
}

export interface PetOverlayHeard {
  id: number
  text: string
  error?: boolean
}

export type PetOverlayControl =
  | { type: 'pop-in' }
  | { type: 'ready' }
  | { type: 'submit'; text: string }
  | { type: 'dictate'; dataUrl: string; mime: string }
  | { type: 'new-chat' }
  | { type: 'mark-read' }
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
const OVERLAY_PAD_X = 260
const OVERLAY_PAD_Y = 420
const OVERLAY_MIN_W = 360
const OVERLAY_MIN_H = 520

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
let dictateHandler: ((audio: Blob) => Promise<string>) | null = null
let newChatHandler: (() => void) | null = null
let lastHeard: PetOverlayHeard | null = null


const THREAD_MAX_TURNS = 12
const THREAD_TURN_MAX_CHARS = 1200

function recentThread(): PetOverlayTurn[] {
  // One question → one answer: mid-turn narration ("Vou verificar…") and
  // tool-round chatter are dropped; each user turn keeps only the LAST
  // assistant text that followed it. While the turn is still running there is
  // no answer yet, and the balloon shows "Pensando…" instead.
  const messages = PRIMARY_SESSION_VIEW.$messages.get()
  const turns: PetOverlayTurn[] = []
  let pendingAnswer: PetOverlayTurn | null = null

  const clip = (text: string) =>
    text.length > THREAD_TURN_MAX_CHARS ? `${text.slice(0, THREAD_TURN_MAX_CHARS).trimEnd()}…` : text

  for (const message of messages) {
    if (message.hidden) {
      continue
    }

    const text = chatMessageText(message).trim()

    if (message.role === 'user') {
      if (pendingAnswer) {
        turns.push(pendingAnswer)
        pendingAnswer = null
      }

      if (text) {
        turns.push({ id: message.id, role: 'user', text: clip(text) })
      }
    } else if (message.role === 'assistant' && text) {
      pendingAnswer = { id: message.id, role: 'assistant', text: clip(text) }
    }
  }

  if (pendingAnswer && !PRIMARY_SESSION_VIEW.$busy.get()) {
    turns.push(pendingAnswer)
  }

  return turns.slice(-THREAD_MAX_TURNS)
}


function currentPayload(): PetOverlayStatePayload {
  return {
    info: $petInfo.get(),
    activity: $petActivity.get(),
    busy: PRIMARY_SESSION_VIEW.$busy.get(),
    awaiting: PRIMARY_SESSION_VIEW.$awaitingResponse.get(),
    unread: $petUnread.get(),
    reaction: $petReaction.get(),
    thread: recentThread(),
    heard: lastHeard
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
    PRIMARY_SESSION_VIEW.$busy.subscribe(pushNow),
    PRIMARY_SESSION_VIEW.$awaitingResponse.subscribe(pushNow),
    $petUnread.subscribe(pushNow),
    $petReaction.subscribe(pushNow),
    PRIMARY_SESSION_VIEW.$messages.subscribe(pushNow)
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

/** Register the handler that starts a fresh conversation from the overlay. */
export function setPetOverlayNewChatHandler(fn: (() => void) | null): void {
  newChatHandler = fn
}

/** Register the handler that turns overlay dictation audio into text. */
export function setPetOverlayDictateHandler(fn: ((audio: Blob) => Promise<string>) | null): void {
  dictateHandler = fn
}

function setHeard(text: string, error = false): void {
  lastHeard = { error, id: (lastHeard?.id ?? 0) + 1, text }
  pushNow()
}

/** Overlay dictation: transcribe through the app's STT path, echo what was
 *  heard into the balloon, then send it like a typed message. */
async function handleDictation(dataUrl: string): Promise<void> {
  if (!dictateHandler) {
    setHeard('Dictation needs the Hermes window open.', true)

    return
  }

  try {
    const audio = await (await fetch(dataUrl)).blob()
    const text = (await dictateHandler(audio)).trim()

    if (!text) {
      setHeard("Didn't catch that. Try again.", true)

      return
    }

    setHeard(text)
    submitHandler?.(text)
  } catch (error) {
    setHeard(error instanceof Error && error.message ? error.message : 'Transcription failed.', true)
  }
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
    } else if (payload?.type === 'mark-read') {
      // Envelope opened in the companion balloon instead of the app.
      clearPetUnread()
    } else if (payload?.type === 'new-chat') {
      lastHeard = null
      newChatHandler?.()
      pushNow()
    } else if (payload?.type === 'dictate' && typeof payload.dataUrl === 'string') {
      void handleDictation(payload.dataUrl)
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
