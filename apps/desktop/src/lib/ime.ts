/**
 * IME-aware Enter handling, shared by every text field whose bare Enter
 * performs an action (submit, rename, commit, adopt, …).
 *
 * CJK/IME users press Enter to *commit a composition* — the candidate text
 * they are still assembling — and that keystroke must never double as the
 * field's submit shortcut. Browsers signal an in-flight composition two ways:
 *
 * - `isComposing` on the (native) keyboard event — the standard signal.
 * - legacy `keyCode === 229` (VK_PROCESSKEY) — Chromium and Safari still
 *   stamp it on keydowns at composition boundaries, including the commit
 *   Enter that can arrive *after* `compositionend` with `isComposing`
 *   already false.
 *
 * One predicate owns that policy so call sites can't drift apart
 * (the main chat composer keeps its own richer stale-flag handling in
 * `app/chat/composer/index.tsx`; everything simpler belongs here).
 *
 * Accepts both React synthetic events (composition state lives on
 * `nativeEvent`) and plain DOM `KeyboardEvent`s (state lives on the event
 * itself), so window-level listeners can share the policy too.
 */
export interface ImeAwareKeyEvent {
  key: string
  isComposing?: boolean
  keyCode?: number
  nativeEvent?: {
    isComposing?: boolean
    keyCode?: number
  }
}

/** Whether this keyboard event belongs to an active IME composition. */
export function isImeComposing(event: ImeAwareKeyEvent): boolean {
  const native = event.nativeEvent ?? event

  return Boolean(native.isComposing || event.isComposing) || native.keyCode === 229 || event.keyCode === 229
}

/** Enter pressed as a real submit — not an IME composition commit. */
export function isSubmitEnter(event: ImeAwareKeyEvent): boolean {
  return event.key === 'Enter' && !isImeComposing(event)
}

/**
 * How long after `compositionend` a bare Enter is still read as the IME's
 * candidate commit rather than the user's send.
 *
 * The two flags above are the first line of defence, and they are not always
 * enough: the commit Enter is delivered *after* `compositionend`, and some
 * IME/engine combinations deliver it with `isComposing` already false and no
 * 229 stamp — so nothing in the event itself says "this keystroke confirmed a
 * candidate". The keystroke's TIMING does: a commit Enter lands within a few
 * tens of milliseconds of the composition ending, while a deliberate send
 * arrives after the user has looked at the text they just committed.
 *
 * 200 ms is deliberately short — long enough to cover the commit keystroke and
 * any IME that re-dispatches it, far short of a human deciding to send. The
 * cost of being wrong is one swallowed Enter inside a composition, which the
 * user recovers by pressing it again.
 */
export const IME_COMMIT_GUARD_MS = 200

/** Whether a bare Enter is close enough to `compositionend` to be that
 *  composition's commit rather than a send. `lastCompositionEndAt` is 0 before
 *  the field has ever composed. */
export function isPostCompositionCommitEnter(lastCompositionEndAt: number, now: number): boolean {
  return lastCompositionEndAt > 0 && now - lastCompositionEndAt <= IME_COMMIT_GUARD_MS
}
