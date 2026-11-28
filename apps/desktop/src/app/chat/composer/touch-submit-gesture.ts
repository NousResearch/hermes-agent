/**
 * The touch press/release gesture that owns Send.
 *
 * Submitting on `pointerdown` fixed the dead button (the press blurred the
 * editor, the keyboard closed and the reflow moved the button out from under the
 * finger, so Chromium never dispatched the click) — but it introduced two bugs:
 *
 *   - a press that drags off the button still sent, because the send had already
 *     happened on the way down; and
 *   - a press held longer than the click-swallow window submitted a second time
 *     on release. By then the draft was empty and the turn was busy, so the
 *     button — now the Stop button — cancelled the turn it had just started.
 *
 * So: the press only ARMS. The release inside the button submits once, and the
 * synthetic click that follows is dropped by a flag that the click itself (or
 * the next press) clears — never by a timer, which is what the hold-out broke.
 */
export function createTouchSubmitGesture() {
  let pressed = false // a finger is down on the button
  let armed = false // releasing here submits
  let swallow = false // the click that follows a handled press must not submit

  return {
    /** Pointer down on the button: arm, and clear any stale swallow. */
    down(): void {
      pressed = true
      armed = true
      swallow = false
    },

    /** Dragged back onto the button mid-press: release would submit again. */
    enter(): void {
      if (pressed) {
        armed = true
      }
    },

    /** Dragged off the button: releasing must not submit. */
    leave(): void {
      if (pressed) {
        armed = false
      }
    },

    /** The gesture was cancelled outright (scroll, system gesture, call). */
    cancel(): void {
      pressed = false
      armed = false
      swallow = true
    },

    /** Pointer up. Returns true when this release is the press's submission.
     *  Either way the click that follows belongs to this gesture, not a send. */
    up(): boolean {
      if (!pressed) {
        return false
      }

      const submit = armed

      pressed = false
      armed = false
      swallow = true

      return submit
    },

    /** Read-and-clear: true when this click must be swallowed. */
    takeClick(): boolean {
      const take = swallow

      swallow = false

      return take
    },
  }
}

export type TouchSubmitGesture = ReturnType<typeof createTouchSubmitGesture>
