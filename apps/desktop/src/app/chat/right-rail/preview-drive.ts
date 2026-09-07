/**
 * PREVIEW DRIVE — the agent's hand on the mouse and keyboard.
 *
 * Everything here goes out through `sendInputEvent`, so the guest page receives
 * ordinary trusted input: the pointer really moves across it, `:hover` really
 * matches, focus really lands, and a menu that only exists while hovered is
 * actually open by the time the click arrives. That last part is the whole
 * point — a synthetic `el.click()` on a dropdown item fires at a node the page
 * never rendered, which is why script-driven clicking quietly misses.
 *
 * The pointer is walked to its target in steps rather than teleported, for the
 * same reason: a page learns it is hovered from a stream of moves, and one
 * move from nowhere to the target skips every element in between. Fourteen
 * steps over ~280ms is what Playwright's action cursor settles on too.
 */

import type { PreviewInputHandle } from './preview-input'

export interface DrivePoint {
  x: number
  y: number
}

/** The pointer JUMPS. These few steps over a couple of frames exist only so the
 *  page receives a move stream at all — `:hover`, `pointerenter` and the menus
 *  that hang off them need more than a single teleport to resolve — not so the
 *  travel can be watched. A machine does not sweep a mouse across a screen. */
const GLIDE_STEPS = 3
const GLIDE_MS = 36

/** Gap between keystrokes. Fast enough not to pad the turn, slow enough that a
 *  page debouncing its input handler still sees them as separate. */
const KEY_MS = 7

/** Notches in a wheel gesture, and how long the gesture takes. Sent as a stream
 *  rather than one big delta so scroll-linked effects — sticky headers, reveal
 *  animations, infinite-scroll loaders — get the same series of events a hand
 *  would produce, and so it looks like momentum rather than a jump cut. */
const WHEEL_STEPS = 12
const WHEEL_MS = 240

/** Where the pointer was left, so the next glide starts from there instead of
 *  jumping. Module-level because the pointer is a property of the pane, not of
 *  any one action. The origin is a placeholder, not a position — see below. */
let pointer: DrivePoint = { x: 0, y: 0 }
let placed = false

/** Whether the pointer has ever been sent anywhere. Until it has, its notional
 *  spot is the top-left corner, which is a real coordinate a caller can wheel
 *  or click at by accident — so callers that only need it to be SOMEWHERE
 *  sensible have to know the difference. */
export function pointerPlaced(): boolean {
  return placed
}

const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

/** Decelerating, like a hand arriving at a target rather than a linear sweep. */
const easeOut = (t: number) => 1 - Math.pow(1 - t, 3)

/** Walk the pointer to `to`, letting the page hover everything on the way. */
export async function glideTo(input: PreviewInputHandle, to: DrivePoint): Promise<void> {
  const from = pointer

  for (let step = 1; step <= GLIDE_STEPS; step++) {
    const progress = easeOut(step / GLIDE_STEPS)

    input.send({
      type: 'mouseMove',
      x: Math.round(from.x + (to.x - from.x) * progress),
      y: Math.round(from.y + (to.y - from.y) * progress)
    })
    await wait(GLIDE_MS / GLIDE_STEPS)
  }

  pointer = to
  placed = true
}

/** Press and release at the pointer's current spot. `clicks` of 3 selects the
 *  text under it, which is how a field gets cleared without a modifier key. */
export async function clickAt(input: PreviewInputHandle, clicks = 1): Promise<void> {
  for (let click = 1; click <= clicks; click++) {
    input.send({ button: 'left', clickCount: click, type: 'mouseDown', x: pointer.x, y: pointer.y })
    input.send({ button: 'left', clickCount: click, type: 'mouseUp', x: pointer.x, y: pointer.y })
    await wait(KEY_MS)
  }
}

/** Wheel `down` pixels' worth of notches at the pointer's current spot; negative
 *  scrolls up. Electron's wheel delta is the legacy `wheelDelta` sign — positive
 *  moves the content down, i.e. scrolls UP — so it is the inverse of the number
 *  a caller asks for, and of DOM `WheelEvent.deltaY`. */
export async function wheelBy(input: PreviewInputHandle, down: number): Promise<void> {
  let sent = 0

  for (let step = 1; step <= WHEEL_STEPS; step++) {
    const so_far = Math.round(down * easeOut(step / WHEEL_STEPS))
    const notch = so_far - sent

    sent = so_far

    if (notch) {
      input.send({ deltaX: 0, deltaY: -notch, type: 'mouseWheel', x: pointer.x, y: pointer.y })
    }

    await wait(WHEEL_MS / WHEEL_STEPS)
  }
}

/** Send one key the long way round, so a page watching any of the three sees it. */
export async function pressKey(input: PreviewInputHandle, key: string): Promise<void> {
  input.send({ keyCode: key, type: 'keyDown' })
  input.send({ keyCode: key, type: 'char' })
  input.send({ keyCode: key, type: 'keyUp' })
  await wait(KEY_MS)
}

/** Select-all inside whatever has focus, which for a focused field is that
 *  field's own text and nothing else. This replaced a triple-click: a triple
 *  click is a POINTER gesture, so it selects whatever paragraph sits under the
 *  cursor whenever the target turns out not to be a field, and the agent was
 *  leaving pages with their body text highlighted. There is no `char` phase —
 *  a chord is not text entry, and sending one types a literal 'a'. */
export async function selectAll(input: PreviewInputHandle): Promise<void> {
  const chord = ['control', 'meta']

  input.send({ keyCode: 'a', modifiers: chord, type: 'keyDown' })
  input.send({ keyCode: 'a', modifiers: chord, type: 'keyUp' })
  await wait(KEY_MS)
}

/** Type `text` a character at a time into whatever currently has focus. */
export async function typeText(input: PreviewInputHandle, text: string): Promise<void> {
  for (const character of text) {
    await pressKey(input, character)
  }
}
