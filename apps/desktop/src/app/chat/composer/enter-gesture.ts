import type { SendGraceReason } from '@/store/composer-prefs'

/** What the composer knows about the press, with no DOM and no timers. */
export interface EnterPressContext {
  /** Is this press inside the double-tap window of the previous one? */
  doubleTap: boolean
  /** Has the typing pause reached `typingIdleMs`? */
  pausedEnough: boolean
  sendOnDoubleTap: boolean
  sendOnHold: boolean
  sendOnPause: boolean
}

/**
 * Who decides a bare Enter that does not send on its own.
 *
 * - `doubleTap`     the press completes the deliberate gesture
 * - `pause`         the pause rule commits, now, on the press
 * - `pauseOnRelease` the pause rule commits, but only once the key comes up
 * - `none`          no rule claims it, so the press lands in the composer
 */
export type EnterPressOwner = 'doubleTap' | 'pause' | 'pauseOnRelease' | 'none'

/**
 * Which rule owns the press.
 *
 * Precedence is the whole point, and it is the one thing the composer and its
 * tests must agree on: **a deliberate gesture outranks the pause rule.** The
 * pause is the app's guess about a press that is not part of a gesture, so it
 * may never decide a press that is. Deciding otherwise made a double tap after
 * a pause take the pause's grace window — a delay on a send the user asked for
 * by hand, which is the one thing the whole feature exists to avoid.
 *
 * That is also why the pause waits for the release when the hold is armed
 * (`pauseOnRelease`): the press in progress can still become a hold, and a hold
 * that is not configured to wait sends at once, so committing on the press
 * would send the message the hold was about to send.
 */
export function composerEnterPressOwner({
  doubleTap,
  pausedEnough,
  sendOnDoubleTap,
  sendOnHold,
  sendOnPause
}: EnterPressContext): EnterPressOwner {
  if (sendOnDoubleTap && doubleTap) {
    return 'doubleTap'
  }

  if (sendOnPause && pausedEnough) {
    return sendOnHold ? 'pauseOnRelease' : 'pause'
  }

  return 'none'
}

/** Is this press inside the double-tap window of the previous one? */
export function isComposerDoubleTap(now: number, lastEnterAt: number, doubleEnterMs: number): boolean {
  return now - lastEnterAt <= doubleEnterMs
}

/** Does a send that arrived this way wait for the grace window? */
export function composerSendDelays(reason: SendGraceReason, graceFor: readonly SendGraceReason[]): boolean {
  return graceFor.includes(reason)
}
