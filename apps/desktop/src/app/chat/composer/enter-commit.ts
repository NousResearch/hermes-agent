import type { KeyboardEvent, RefObject } from 'react'

import { triggerHaptic } from '@/lib/haptics'
import { isPostCompositionCommitEnter } from '@/lib/ime'
import { activeSendGestures, type ComposerSendPrefs, type SendGraceReason } from '@/store/composer-prefs'
import type { QueuedPromptEntry } from '@/store/composer-queue'

import { liveComposerDraft, type QueueEditState } from './composer-utils'
import { composerEnterPressOwner, composerSendDelays, isComposerDoubleTap } from './enter-gesture'
import { resolveComposerEnterKeyIntent } from './enter-key-mode'
import type { ComposerSendGrace } from './hooks/use-composer-send-grace'
import { composerPlainText, renderComposerContents } from './rich-editor'

/**
 * The Enter key: what a press means, and what its release still owes.
 *
 * This is the composer's send-gesture decision tree, lifted out of ChatBar so its
 * keydown handler keeps only the keys it owns (undo/redo, the trigger popover,
 * the arrows, the directive chips). Nothing here holds composer state: the caller
 * passes the live editor, the draft, its own callbacks and the press record the
 * gesture hook owns, and this module decides — in ONE fixed order — which of them
 * acts.
 *
 * `handleComposerEnterKeyDown` runs AFTER the popover, history and chip handlers
 * have had their turn, so an Enter that accepts a completion never reaches it. It
 * runs BEFORE the editor's default, and says so itself by calling
 * `preventDefault` on exactly the presses it owns: an unprevented press is a line
 * break the editor was meant to insert.
 */

/** What the release should do, when the press in progress did not become a
 *  gesture: run the pause rule's send, or commit a send already waiting on the
 *  grace window. */
export type PendingPress = 'pause' | 'commit' | null

/**
 * Everything a press leaves behind, in one record per composer instance.
 *
 * One record rather than eight separate refs: the keydown, the keyup, the hold
 * timer and the idle deadline all read and write these together, and the gesture
 * module takes them as one explicit dependency instead of reaching for them.
 * Mutated in place — none of it may paint.
 */
export interface ComposerPressState {
  /** When the last composition ended, for the timing guard that catches an IME's
   *  candidate-confirm Enter arriving after `compositionend` with no flag and no
   *  keyCode 229 left on it (see lib/ime.ts). */
  compositionEndedAt: number
  /** The `hold` mode's threshold timer. Owned by the press (keydown starts it,
   *  keyup cancels it) so the gesture is decided by a value we control rather
   *  than by the operating system's key-repeat delay. */
  holdTimer: number | undefined
  /** The idle-send deadline: armed on every edit, cleared by the next one. There
   *  is no key to watch, so the timer IS the whole gesture. */
  idleTimer: number | undefined
  /** Timestamp of the last plain Enter. A second press inside the double-tap
   *  window commits the draft; a lone press just breaks the line. */
  lastEnterAt: number
  /** Timestamp of the last plain Shift+Enter, for the steer chord's own double
   *  press. A separate stamp from `lastEnterAt` so a send pair and a steer pair
   *  cannot complete each other. */
  lastSteerAt: number
  /** Whatever the press deferred to its release (see `PendingPress`). */
  pending: PendingPress
  /** Last keystroke in the editor, for the `pause` mode's "have you stopped?"
   *  test. Seeded at 0 so an untouched composer counts as idle. */
  typedAt: number
  /** Whether a real CHARACTER key has been pressed since the composer last held
   *  content. The idle send requires it, so a pasted blob or a restored draft can
   *  never send itself before the user has touched it. */
  typedSinceContent: boolean
}

/** A fresh press record for a composer that is mounting. */
export function createComposerPressState(): ComposerPressState {
  return {
    compositionEndedAt: 0,
    holdTimer: undefined,
    idleTimer: undefined,
    lastEnterAt: 0,
    lastSteerAt: 0,
    pending: null,
    typedAt: 0,
    typedSinceContent: false
  }
}

/** What the Enter path may do to the composer, owned by the gesture hook. */
export interface ComposerEnterGestures {
  /** Stop the hold threshold timer before it fires: the press was a tap. */
  cancelEnterHold: () => void
  /** Commit the draft a long press has been holding. */
  commitHeldEnter: () => void
  /** The waiting send, and the timers it shares with this module. */
  press: ComposerPressState
  sendGrace: ComposerSendGrace
}

/**
 * Everything the Enter path reads, passed explicitly: no React context, no
 * store, no module-level state. The composer builds this at press time, so every
 * value is the one the live editor and the live queue have right now.
 */
export interface ComposerEnterDeps {
  /** Live attachments. A send with only an image still has a payload. */
  attachmentCount: number
  busy: boolean
  /** Whether a steer can reach the running turn right now. */
  canSteer: boolean
  /** Deliver the queue's head into the running turn (busy Enter on an empty draft). */
  deliverQueuedNow: (id: string) => Promise<unknown>
  disabled: boolean
  /** The draft mirror the send paths read after the editor. */
  draftRef: RefObject<string>
  drainNextQueued: () => Promise<boolean>
  editorRef: RefObject<HTMLDivElement | null>
  /** Pull the live editor text into draft state before a send reads it. */
  flushEditorToDraft: (editor: HTMLDivElement) => void
  gestures: ComposerEnterGestures
  /** The send settings, read once at the moment of the press (see the keydown). */
  prefs: ComposerSendPrefs
  /** Commit the draft as a follow-up while a turn runs. */
  queueDraft: () => void
  queueEdit: QueueEditState | null
  queuedPrompts: readonly QueuedPromptEntry[]
  setComposerText: (value: string) => void
  /** Redirect the running turn with the draft. */
  steerDraft: () => void
  submitDraft: () => void
}

/**
 * Drop the break the press just inserted, so a send carries no trailing newline.
 * Nothing to drop when the press does not break the line in the first place — the
 * callers check that, because only they know what the press was asked to do.
 */
export function dropTrailingBreak(editorRef: RefObject<HTMLDivElement | null>): void {
  const editor = editorRef.current

  if (!editor) {
    return
  }

  const inserted = composerPlainText(editor)

  if (inserted.endsWith('\n')) {
    renderComposerContents(editor, inserted.replace(/\n+$/, ''))
  }
}

/**
 * Is this Enter an IME's commit rather than a send?
 *
 * The composition flags on the event are the first line of defence (see the
 * composer's own stale-flag heal and its isComposing guard). These are the two
 * cases they miss.
 */
export function isImeCommitEnter(
  event: KeyboardEvent<HTMLDivElement>,
  compositionEndedAt: number,
  now: number
): boolean {
  if (event.key !== 'Enter') {
    return false
  }

  // macOS Chinese IME (and some 3rd-party IMEs on Windows) emit Enter with
  // keyCode 229 (legacy VK_PROCESSKEY) while isComposing is already false. The
  // compositionend has fired but the keydown still carries 229, signalling "this
  // Enter is an IME commit, not a user send". If we let it through, the message
  // fires before the committed text is fully in the DOM.
  if (event.keyCode === 229) {
    return true
  }

  // An IME/engine combination that delivers the candidate-confirm Enter after
  // `compositionend` with isComposing false and no 229 — nothing in the event
  // names it a commit, so both flags pass and the message sends with the
  // candidate the user was still picking (#49422: "after pinyin/candidate
  // selection, pressing Enter to confirm actually triggers send"). Only the
  // timing says what it is: a commit Enter lands immediately after the
  // composition ends, while a send comes after the user has read the text they
  // just committed. Bare Enter only — ⌘/Ctrl+Enter is an explicit send and
  // Shift+Enter never sends, so neither should be swallowed here.
  if (event.shiftKey || event.metaKey || event.ctrlKey) {
    return false
  }

  return isPostCompositionCommitEnter(compositionEndedAt, now)
}

/** Does a send started THIS way wait for the grace window? */
function delays(prefs: ComposerSendPrefs, reason: SendGraceReason): boolean {
  return composerSendDelays(reason, prefs.sendGraceFor)
}

/**
 * An Enter press, once no earlier handler has taken it (see the module note).
 *
 * Reads the send prefs ONCE, at press time: none of this may change mid-gesture
 * (a settings change while the key is held would otherwise make the repeat that
 * arrives a few ms later mean something else), and the branches below would
 * otherwise disagree about the same press.
 */
export function handleComposerEnterKeyDown(event: KeyboardEvent<HTMLDivElement>, deps: ComposerEnterDeps): void {
  const { prefs } = deps

  if (event.repeat) {
    repeatedEnter(event, prefs)

    return
  }

  // Cmd/Ctrl+Enter commits the draft unconditionally: it queues a follow-up while
  // a turn runs, and sends outright when the session is idle. So the chord means
  // "send" in both states, and the plain-Enter double-tap is a convenience, not
  // the only way out.
  if ((event.metaKey || event.ctrlKey) && !event.shiftKey) {
    commitEnterChord(event, deps)

    return
  }

  // Shift+Enter steers a running turn: the multiline-first binding the shortcuts
  // panel, the settings description and the user guide all promise (Settings →
  // Keyboard Shortcuts → Send behavior). Only while a turn is actually steerable
  // — otherwise the chord keeps its native meaning and breaks the line, which is
  // what a bare Enter already does in this mode.
  if (!prefs.enterSends && event.shiftKey && deps.canSteer) {
    steerEnterChord(event, deps, prefs)

    return
  }

  // Plain Enter. What it does depends on the send settings (Settings → Keyboard
  // Shortcuts → Send behavior): it commits the draft, breaks the line, or commits
  // on the second tap. See store/composer-prefs.ts.
  if (!event.shiftKey) {
    plainEnter(event, deps, prefs)
  }
}

/** What the release owes the press it ended (see `PendingPress`). */
export function releaseComposerEnterKey(deps: ComposerEnterDeps): void {
  const { cancelEnterHold, press, sendGrace } = deps.gestures

  // Releasing Enter before the hold timer fires means it was a tap: whatever the
  // press put in the composer stands, and nothing is sent. The timer is the ONLY
  // thing that can turn a press into a send — with one exception: a press that
  // was deferred to this release because the hold was armed and it could still
  // have become one. A tap is not a hold, so what the press deferred runs now.
  cancelEnterHold()

  const deferred = press.pending

  press.pending = null

  if (deferred === 'pause') {
    // The pause rule gets its turn after all, with its own grace window if the
    // user asked for one.
    if (delays(deps.prefs, 'pause') && sendGrace.hold()) {
      triggerHaptic('submit')
    } else {
      deps.submitDraft()
    }
  } else if (deferred === 'commit' && sendGrace.holding) {
    // Only while the send is still waiting: the window may have fired while the
    // key was down, in which case it has already committed.
    sendGrace.cancel()
    commitIfPayload(deps)
  }
}

/** A held key REPEATS, and a repeat is not a press. */
function repeatedEnter(event: KeyboardEvent<HTMLDivElement>, prefs: ComposerSendPrefs): void {
  // Auto-repeat produces a stream of bare Enter keydowns that would otherwise
  // complete a double-tap (the same key twice inside the window is
  // indistinguishable from a deliberate gesture), drain the queue a second time,
  // or start a hold — so holding Enter to add blank lines would commit the draft
  // instead.
  //
  // `hold` is the mode where the long press IS the gesture, but even there the
  // repeat is not what decides: its own timer is (started by the press), so the
  // threshold is a value we control rather than whatever the operating system's
  // key repeat happens to be — and it still works for anyone with repeat
  // switched off. So a repeat never reaches the gesture logic.
  //
  // It still falls through UNPREVENTED when a bare Enter only breaks the line and
  // no gesture owns it, so holding the key keeps inserting them; a mode where
  // Enter sends, or where any gesture is armed, gives a repeat no meaning and it
  // is swallowed rather than leaving stray breaks behind.
  if (prefs.enterSends || activeSendGestures(prefs).length > 0) {
    event.preventDefault()
  }
}

/** Cmd/Ctrl+Enter: commit now, queueing a follow-up when a turn is running. */
function commitEnterChord(event: KeyboardEvent<HTMLDivElement>, deps: ComposerEnterDeps): void {
  event.preventDefault()
  deps.gestures.press.lastEnterAt = 0

  if (deps.disabled) {
    return
  }

  if (deps.busy) {
    // As with plain Enter, source the just-typed content from the DOM so a fast
    // keypress cannot queue a stale draft.
    const editorText = liveComposerDraft(deps.editorRef.current, deps.draftRef.current)

    if (editorText !== deps.draftRef.current) {
      deps.draftRef.current = editorText
      deps.setComposerText(editorText)
    }

    deps.queueDraft()

    return
  }

  deps.submitDraft()
}

/** Shift+Enter, while a turn is steerable: redirect it (or arm the double press). */
function steerEnterChord(
  event: KeyboardEvent<HTMLDivElement>,
  deps: ComposerEnterDeps,
  prefs: ComposerSendPrefs
): void {
  const { press } = deps.gestures

  // With the double tap armed the chord takes the same guard as every other send:
  // two presses. Shift is held for capitals, so a lone Shift+Enter is exactly the
  // press that arrives by accident, and it must not redirect a live turn.
  if (prefs.sendOnDoubleTap) {
    const steeredAt = Date.now()

    if (steeredAt - press.lastSteerAt > prefs.doubleEnterMs) {
      press.lastSteerAt = steeredAt

      // The first press of the pair. It breaks the line only when the user kept
      // the line break on, the same rule a bare press follows.
      if (!prefs.enterNewline) {
        event.preventDefault()
      }

      return
    }

    press.lastSteerAt = 0
  }

  event.preventDefault()

  // Drop the break the first press of the pair inserted, the way the double-tap
  // send does, so a steered message carries no trailing newline.
  if (prefs.enterNewline) {
    dropTrailingBreak(deps.editorRef)
  }

  // Source the just-typed text from the DOM before redirecting, so a fast keypress
  // cannot steer a stale draft.
  deps.flushEditorToDraft(event.currentTarget)
  deps.steerDraft()
}

/** A bare Enter: drain, deliver, commit, or hand the press to the gestures. */
function plainEnter(event: KeyboardEvent<HTMLDivElement>, deps: ComposerEnterDeps, prefs: ComposerSendPrefs): void {
  // Resolve Enter from the live DOM, not render-derived composer state.
  const editorText = liveComposerDraft(deps.editorRef.current, deps.draftRef.current)
  // Decide from the DOM, not React state. `hasComposerPayload` is derived from the
  // AUI composer state, which lags the latest keystroke by a render, so on fast
  // typing / IME the just-typed text isn't in state yet. Without the live read, a
  // real message typed while prompts are queued would drain the queue instead of
  // sending. submitDraft() re-syncs and sends the live editor text.
  const hasLivePayload = editorText.trim().length > 0 || deps.attachmentCount > 0

  if (deps.disabled) {
    return
  }

  // Empty Enter: drain the queue when idle, deliver its head while busy. With
  // nothing typed there is no line to break and no draft to commit, so these
  // gestures keep their single press in every send mode.
  if (!hasLivePayload) {
    emptyEnter(event, deps)

    return
  }

  // A bare Enter that commits needs no other branch.
  if (prefs.enterSends) {
    event.preventDefault()

    if (delays(prefs, 'enter') && deps.gestures.sendGrace.hold()) {
      triggerHaptic('submit')

      return
    }

    deps.submitDraft()

    return
  }

  gesturesEnter(event, deps, prefs)
}

/** Empty Enter: drain the queue when idle, deliver its head while busy. */
function emptyEnter(event: KeyboardEvent<HTMLDivElement>, deps: ComposerEnterDeps): void {
  event.preventDefault()
  deps.gestures.press.lastEnterAt = 0

  if (!deps.busy && deps.queuedPrompts.length > 0) {
    void deps.drainNextQueued()

    return
  }

  if (!deps.busy) {
    return
  }

  // While busy, a second Enter delivers the queue's head now: steered into the
  // live turn when a steer can carry it, else promoted and interrupted on settle.
  // With nothing queued it stays a no-op, because interrupting is explicit
  // (Stop/Esc), never a stray Enter after sending.
  const head = deps.queuedPrompts.find(entry => entry.id !== deps.queueEdit?.entryId)

  if (head) {
    void deps.deliverQueuedNow(head.id)
  }
}

/**
 * The gesture ladder for a press that does not send on its own.
 *
 * Fixed order: double tap, then hold, then the pause rule. A deliberate gesture
 * always outranks the guess — the pause describes a press that is not part of
 * one, so it may never decide a press that is. Every branch reads the same
 * double-tap window, and the press is stamped before any branch can return.
 */
function gesturesEnter(event: KeyboardEvent<HTMLDivElement>, deps: ComposerEnterDeps, prefs: ComposerSendPrefs): void {
  const { cancelEnterHold, commitHeldEnter, press } = deps.gestures
  const now = Date.now()
  const doubleTap = isComposerDoubleTap(now, press.lastEnterAt, prefs.doubleEnterMs)

  press.lastEnterAt = now

  const pausedEnough = prefs.sendOnPause && now - press.typedAt > prefs.typingIdleMs

  const owner = composerEnterPressOwner({
    doubleTap,
    pausedEnough,
    sendOnDoubleTap: prefs.sendOnDoubleTap,
    sendOnHold: prefs.sendOnHold,
    sendOnPause: prefs.sendOnPause
  })

  cancelEnterHold()

  // The hold timer starts even on a press the pause rule has an opinion about. A
  // long press is an additional way out rather than a different opinion about the
  // tap, so it has to stay reachable; the pause send waits for the release instead
  // (see `pending`), because a press that can still become a hold must not have
  // sent already.
  if (prefs.sendOnHold) {
    press.holdTimer = window.setTimeout(() => {
      press.holdTimer = undefined
      commitHeldEnter()
    }, prefs.holdMs)
  }

  if (owner === 'doubleTap') {
    doubleTapEnter(event, deps, prefs)

    return
  }

  if (prefs.commitOnPress && deps.gestures.sendGrace.holding) {
    commitWaitingSend(event, deps, prefs)

    return
  }

  // `pause` with the hold armed: the commit moves to the release, because the
  // press in progress can still become a hold, and a hold that is not configured
  // to wait sends at once — committing here would send the message the hold was
  // about to send.
  if (owner === 'pauseOnRelease') {
    event.preventDefault()
    press.pending = 'pause'

    return
  }

  if (owner === 'pause') {
    pauseEnter(event, deps, prefs)

    return
  }

  unclaimedEnter(event, deps, prefs)
}

/** The second of two fast presses: commit, taking back any open grace window. */
function doubleTapEnter(event: KeyboardEvent<HTMLDivElement>, deps: ComposerEnterDeps, prefs: ComposerSendPrefs): void {
  const { cancelEnterHold, press, sendGrace } = deps.gestures

  event.preventDefault()

  // This press supersedes whatever the previous one decided, including a pause
  // send that is still waiting out its grace window.
  press.pending = null

  // The line break lands immediately rather than on a timer, so text typed right
  // after Enter can never land on the wrong side of it — drop the break the first
  // press just inserted, otherwise every double-tap message would ship with a
  // trailing newline. Nothing to drop when the press does not break the line in
  // the first place.
  if (prefs.enterNewline) {
    dropTrailingBreak(deps.editorRef)
  }

  cancelEnterHold()

  // The previous press may already have sent this draft, in which case there is
  // nothing left to commit and submitting would drain or steer.
  if (!hasDraftPayload(deps)) {
    return
  }

  if (delays(prefs, 'doubleTap')) {
    if (sendGrace.hold()) {
      triggerHaptic('submit')
    }

    return
  }

  // Not configured to wait, so this press is the user saying "now": commit on the
  // spot and take back any window the previous press opened, or it would fire a
  // second send into an empty composer later.
  sendGrace.cancel()
  deps.submitDraft()
}

/**
 * A press during a wait: the send is already going, so a press that is not part of
 * a gesture is the user saying "now", not "wait longer". Without this the window
 * restarts on every press, sliding the send a full grace period each time.
 */
function commitWaitingSend(
  event: KeyboardEvent<HTMLDivElement>,
  deps: ComposerEnterDeps,
  prefs: ComposerSendPrefs
): void {
  const { press, sendGrace } = deps.gestures

  event.preventDefault()

  // Deferred to the release while the hold is armed, because this press can still
  // become a hold, and the hold owns its own decision — the gestures are never
  // touched by this option.
  if (prefs.sendOnHold) {
    press.pending = 'commit'

    return
  }

  sendGrace.cancel()
  commitIfPayload(deps)
}

/** The single Enter after a typing pause: the send the user meant. */
function pauseEnter(event: KeyboardEvent<HTMLDivElement>, deps: ComposerEnterDeps, prefs: ComposerSendPrefs): void {
  event.preventDefault()

  // Held for the grace window when one is configured, which is the whole reason a
  // guessed send is safe to ship.
  if (delays(prefs, 'pause') && deps.gestures.sendGrace.hold()) {
    triggerHaptic('submit')

    return
  }

  deps.submitDraft()
}

/** No gesture claimed the press: it lands in the composer, and never sends. */
function unclaimedEnter(event: KeyboardEvent<HTMLDivElement>, deps: ComposerEnterDeps, prefs: ComposerSendPrefs): void {
  const intent = resolveComposerEnterKeyIntent({
    enterNewline: prefs.enterNewline,
    enterSends: prefs.enterSends,
    key: event.key
  })

  // No gesture is armed, so the press lands in the composer — or is dropped, when
  // the user asked for a bare Enter that only stays out of the way. Either way it
  // never sends. The mode module owns that question: gestures never get here, so
  // this is the only decision left to make.
  if (!prefs.sendOnDoubleTap) {
    if (intent === 'ignore') {
      event.preventDefault()
    }

    return
  }

  // The first press of a possible pair: fall through UNPREVENTED so the editor
  // inserts the line break, exactly as it does for Shift+Enter — unless the press
  // is meant to do nothing at all.
  if (intent === 'ignore') {
    event.preventDefault()
  }
}

/** Submit, but only while the composer still holds something to send. */
function commitIfPayload(deps: ComposerEnterDeps): void {
  if (hasDraftPayload(deps)) {
    deps.submitDraft()
  }
}

/** Does the live editor (or the attachment tray) still hold a payload? */
function hasDraftPayload(deps: ComposerEnterDeps): boolean {
  const editor = deps.editorRef.current
  const draftNow = editor ? composerPlainText(editor) : ''

  return draftNow.trim().length > 0 || deps.attachmentCount > 0
}
