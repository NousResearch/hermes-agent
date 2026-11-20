import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useState } from 'react'
import type { RefObject } from 'react'

import { triggerHaptic } from '@/lib/haptics'
import { $composerSendPrefs } from '@/store/composer-prefs'

import {
  type ComposerEnterGestures,
  type ComposerPressState,
  createComposerPressState,
  dropTrailingBreak
} from '../enter-commit'
import { composerSendDelays } from '../enter-gesture'
import { composerPlainText } from '../rich-editor'

import { useComposerSendGrace } from './use-composer-send-grace'

interface Options {
  /** A running turn never arms the idle send: queueing and steering are deliberate acts. */
  busy: boolean
  /** The live editor: the payload checks below re-run against it when a timer fires. */
  editorRef: RefObject<HTMLDivElement | null>
  /** The session a waiting send belongs to. A swap takes the window back. */
  ownerKey: string | null
  /** Commit the draft — the same submitDraft the keystroke path calls. */
  submitDraft: () => void
}

/** The press record plus everything that acts on it (see enter-commit.ts). */
export interface ComposerSendGestures extends ComposerEnterGestures {
  /** (Re-)arm the idle deadline from the live editor. */
  armIdleSend: () => void
  cancelIdleSend: () => void
  /** A press in flight is over (focus left the editor): no timer, no deferred send. */
  cancelPress: () => void
  /** A real character key landed: the idle send may count this draft as typed. */
  noteCharacterKey: () => void
  /** A composition ended just now (see the IME timing guard). */
  noteCompositionEnd: () => void
  /** The editor changed: stamp the typing pause, take back a held send, re-arm the idle deadline. */
  noteEditorInput: (empty: boolean) => void
}

/**
 * The composer's send gestures: the grace window, the press record, the hold
 * threshold and the idle deadline.
 *
 * One owner for state that must never paint: every timer here decides a send, and
 * a render on the way through would let a stale closure decide instead. The
 * record is per composer instance, so two composers (main + tile) never share a
 * press, and the module that reads it (enter-commit.ts) receives it as an
 * argument rather than reaching for it.
 */
export function useComposerSendGestures({ busy, editorRef, ownerKey, submitDraft }: Options): ComposerSendGestures {
  // One record per composer instance, stable across renders and mutated in place:
  // the press is never a render concern, so nothing here may paint.
  const [press] = useState<ComposerPressState>(createComposerPressState)

  // The send settings, mirrored from the gateway config: only the grace window's
  // length is a render concern here — every other read happens at press time.
  const sendPrefs = useStore($composerSendPrefs)

  // Holds an inferred send so Esc can take it back. The draft stays in the
  // composer for the whole hold — the commit is the same submitDraft the
  // keystroke would have called, just later. `ownerKey` is the session whose
  // draft a waiting send would commit, so a swap still waiting takes its window
  // back instead of sending the new one's.
  const sendGrace = useComposerSendGrace({
    graceMs: sendPrefs.sendGraceMs,
    onCommit: submitDraft,
    ownerKey
  })

  // Released before the timer fires = a tap, not a hold. Nothing to undo: the
  // break the press inserted is exactly what a tap was supposed to leave.
  const cancelEnterHold = useCallback(() => {
    if (press.holdTimer !== undefined) {
      window.clearTimeout(press.holdTimer)
      press.holdTimer = undefined
    }
  }, [press])

  /** Commit the draft a long press has been holding. Drops the break the press
   *  inserted, so a hold sends the same text a tap would have left in the box,
   *  and goes through the same grace window every other send uses — the hold is
   *  its own situation, so it is delayed only if the user asked for that one. */
  const commitHeldEnter = useCallback(() => {
    // The hold decided this press, so the release must not also run whatever
    // the press deferred.
    press.pending = null

    dropTrailingBreak(editorRef)

    if (composerSendDelays('hold', $composerSendPrefs.get().sendGraceFor) && sendGrace.hold()) {
      triggerHaptic('submit')

      return
    }

    submitDraft()
  }, [editorRef, press, sendGrace, submitDraft])

  const cancelIdleSend = useCallback(() => {
    if (press.idleTimer !== undefined) {
      window.clearTimeout(press.idleTimer)
      press.idleTimer = undefined
    }
  }, [press])

  /** A press in flight belongs to the editor that started it (see the blur
   *  handler): end it outright, so no timer and no deferred send outlive focus. */
  const cancelPress = useCallback(() => {
    cancelEnterHold()
    press.pending = null
  }, [cancelEnterHold, press])

  /** The one send nobody asked for: it takes the grace window ALWAYS, whatever
   *  the per-situation checkboxes say. There is no press to point at, so the
   *  hold is the only thing between a pause for thought and a message the user
   *  never sent. */
  const commitIdleSend = useCallback(() => {
    press.idleTimer = undefined

    // The draft can be gone by now (sent by hand, cleared, session swapped).
    // Submitting an empty composer is not a no-op — it drains or steers — so the
    // payload is re-checked at the moment the timer fires.
    const editor = editorRef.current

    if (!editor || composerPlainText(editor).trim().length === 0) {
      return
    }

    if (sendGrace.hold()) {
      triggerHaptic('submit')

      return
    }

    submitDraft()
  }, [editorRef, press, sendGrace, submitDraft])

  /** Arm (or re-arm) the idle send from the current state. Called on every edit,
   *  so typing pushes the deadline out — a fresh window each time, rather than a
   *  fixed clock a momentary pause mid-thought could trip. */
  const armIdleSend = useCallback(() => {
    cancelIdleSend()

    const { enterSends, idleSendMs, sendOnIdle } = $composerSendPrefs.get()
    const editor = editorRef.current
    const hasPayload = Boolean(editor && composerPlainText(editor).trim().length > 0)

    // Never while a turn is running: queueing and steering are deliberate acts,
    // and the app guessing on top of them is not. Never for content that arrived
    // some other way either — a pasted blob or a restored draft has to be typed
    // into before it can send itself.
    if (!sendOnIdle || enterSends || busy || !hasPayload || !press.typedSinceContent) {
      return
    }

    press.idleTimer = window.setTimeout(commitIdleSend, idleSendMs)
  }, [busy, cancelIdleSend, commitIdleSend, editorRef, press])

  // A composer that unmounts mid-press must not fire a send into a dead tree —
  // and the idle deadline must not outlive the composer that armed it either.
  // A session swap keeps this composer mounted but replaces the send's owner (the
  // draft/queue it would submit), so the swap re-runs this cleanup too: a deadline
  // armed for the old session is cancelled rather than left to submit the new
  // one's draft.
  useEffect(
    () => () => {
      cancelEnterHold()
      cancelIdleSend()
    },
    [cancelEnterHold, cancelIdleSend, ownerKey]
  )

  /** Live typing: the signal `pause` mode reads, and the signal that takes a held
   *  send back — you touched it, so the guess was wrong. An empty composer has
   *  nothing to send itself, and the next content to arrive has to be typed into
   *  before the idle deadline counts again. */
  const noteEditorInput = useCallback(
    (empty: boolean) => {
      press.typedAt = Date.now()
      sendGrace.cancel()

      if (empty) {
        press.typedSinceContent = false
      }

      armIdleSend()
    },
    [armIdleSend, press, sendGrace]
  )

  /** A real character key. Modifiers, arrows and the like do not count: the idle
   *  send asks "did the user type this", not "did they touch the keyboard". */
  const noteCharacterKey = useCallback(() => {
    press.typedSinceContent = true
  }, [press])

  /** Stamp the moment a composition ended, for the IME timing guard. */
  const noteCompositionEnd = useCallback(() => {
    press.compositionEndedAt = Date.now()
  }, [press])

  return {
    armIdleSend,
    cancelEnterHold,
    cancelIdleSend,
    cancelPress,
    commitHeldEnter,
    noteCharacterKey,
    noteCompositionEnd,
    noteEditorInput,
    press,
    sendGrace
  }
}
