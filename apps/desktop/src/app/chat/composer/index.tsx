import { ComposerPrimitive } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type ClipboardEvent, type FormEvent, type KeyboardEvent, useCallback, useEffect, useMemo, useRef } from 'react'

import { composerFill, composerFloatingStrip, composerSurfaceGlass } from '@/components/chat/composer-dock'
import { Button } from '@/components/ui/button'
import { Slot as ContribSlot } from '@/contrib/react/slot'
import { useI18n } from '@/i18n'
import { chatMessageText } from '@/lib/chat-messages'
import { sanitizeComposerInput } from '@/lib/composer-input-sanitize'
import { DATA_IMAGE_URL_RE } from '@/lib/embedded-images'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'
import { interceptsTypedVoiceStop } from '@/lib/voice-stop-word'
import { sessionCompacting } from '@/store/compaction'
import { browseBackward, browseForward, deriveUserHistory, isBrowsingHistory } from '@/store/composer-input-history'
import { POPOUT_WIDTH_REM } from '@/store/composer-popout'
import { parkQueuedPrompts, removeQueuedPrompt, unparkQueuedPrompts } from '@/store/composer-queue'
import { toggleReview } from '@/store/review'
import { $gatewayState } from '@/store/session'
import { $threadScrolledUp } from '@/store/thread-scroll'
import { $autoSpeakReplies } from '@/store/voice-prefs'
import { useTheme } from '@/themes'

import { AttachmentList } from './attachments'
import {
  acceptsTriggerCompletion,
  COMPOSER_FADE_BACKGROUND,
  type QueueEditState,
  slashArgStage
} from './composer-utils'
import { ContextMenu } from './context-menu'
import { COMPOSER_AREAS, runComposerMiddleware } from './contrib'
import { ComposerControls } from './controls'
import { COMPOSER_DROP_ACTIVE_CLASS, COMPOSER_DROP_FADE_CLASS } from './drop-affordance'
import { markActiveComposer } from './focus'
import { HelpHint } from './help-hint'
import { useAtCompletions } from './hooks/use-at-completions'
import { useComposerBranch } from './hooks/use-composer-branch'
import { useComposerDraft } from './hooks/use-composer-draft'
import { useComposerDrop } from './hooks/use-composer-drop'
import { useComposerEscCancel } from './hooks/use-composer-esc-cancel'
import { useComposerMetrics } from './hooks/use-composer-metrics'
import { useComposerPlaceholder } from './hooks/use-composer-placeholder'
import { useComposerPopout } from './hooks/use-composer-popout'
import { useComposerQueue } from './hooks/use-composer-queue'
import { useComposerSubmit } from './hooks/use-composer-submit'
import { useComposerTrigger } from './hooks/use-composer-trigger'
import { useComposerUndo } from './hooks/use-composer-undo'
import { useComposerUrlDialog } from './hooks/use-composer-url-dialog'
import { useComposerVoice } from './hooks/use-composer-voice'
import { useEmojiCompletions } from './hooks/use-emoji-completions'
import { useComposerMicroActions } from './hooks/use-micro-actions'
import { useSlashCompletions } from './hooks/use-slash-completions'
import { useSessionStatusPresence } from './hooks/use-status-presence'
import { ActionBadges } from './micro-actions'
import { chipTypedPathOnSpace, pathifyRefs } from './path-refs'
import { QueuePanel } from './queue-panel'
import {
  COMPOSER_PLACEHOLDER_CLASS,
  composerPlainText,
  deleteChipBeforeCaret,
  deleteSelectionInEditor,
  insertComposerContentsAtCaret,
  normalizeComposerEditorDom,
  RICH_INPUT_SLOT
} from './rich-editor'
import { useComposerScope } from './scope'
import { ComposerStatusStack } from './status-stack'
import { CodingStatusRow } from './status-stack/coding-row'
import { extractClipboardImageBlobs, openDirectiveScope } from './text-utils'
import { ComposerTriggerPopover } from './trigger-popover'
import type { ChatBarProps } from './types'
import { isRedoShortcut, isUndoShortcut } from './undo-history'
import { UrlDialog } from './url-dialog'
import { chipTypedUrlOnSpace, linkifyUrls } from './url-refs'
import { VoiceActivity, VoicePlaybackActivity } from './voice-activity'

export function ChatBar({
  busy,
  cwd,
  disabled,
  focusKey,
  gateway,
  maxRecordingSeconds = 120,
  queueSessionKey,
  sessionId,
  state,
  onCancel,
  onAddUrl,
  onAttachDroppedItems,
  onAttachImageBlob,
  onPasteClipboardImage,
  onPickFiles,
  onPickFolders,
  onPickImages,
  onRemoveAttachment,
  onSteer,
  onSubmit: onSubmitProp,
  onTranscribeAudio
}: ChatBarProps) {
  // Typed stop phrase during an active voice conversation ends it — same
  // semantics as SAYING "stop" (voice-stop-word.ts) or clicking the pill's
  // end control. Populated after useComposerVoice below (the submit wrapper
  // is created first); render-time assignment keeps the ref current.
  const voiceStopRef = useRef<{ active: boolean; end: () => void }>({ active: false, end: () => {} })

  // Every send (typed, queued, voice) passes through the contributed
  // middleware chain first — rewrite / pass-through / cancel. Empty chain =
  // exact pass-through, so surfaces without contributions are byte-identical.
  const onSubmit = useCallback<ChatBarProps['onSubmit']>(
    async (value, options) => {
      // Bare stop phrase typed while the voice conversation is live: end the
      // conversation (mic off, pill dismissed) instead of sending "stop" to
      // the agent. Spoken transcripts are already stop-checked inside
      // use-voice-conversation, so this only catches typed/queued sends.
      // Outside a voice conversation, typed "stop" is a normal message.
      const voiceStop = voiceStopRef.current

      if (interceptsTypedVoiceStop(voiceStop.active, value, options?.attachments?.length ?? 0)) {
        voiceStop.end()

        // Consumed (not rejected): report accepted so the submit engine
        // clears the draft instead of restoring "stop" into the composer.
        return true
      }

      const draft = await runComposerMiddleware({ text: value, attachments: options?.attachments })

      if (!draft) {
        return false
      }

      return onSubmitProp(draft.text, { ...options, attachments: draft.attachments })
    },
    [onSubmitProp]
  )

  // Which live composer this instance IS (main | tile) — its attachment set,
  // focus-bus key, and awaiting-input edge. Main scope = the legacy globals.
  const scope = useComposerScope()
  const attachments = useStore(scope.attachments.$attachments)
  const compacting = useStore(useMemo(() => sessionCompacting(sessionId ?? null), [sessionId]))
  const scrolledUp = useStore($threadScrolledUp)
  const autoSpeak = useStore($autoSpeakReplies)
  // The turn is parked on the user (clarify / approval / sudo / secret). Esc must
  // not interrupt it — there's nothing actively running to stop, and stopping
  // would discard a question the user may want to come back to. The blocking
  // prompt owns its own dismissal (Skip, Reject, dialog close).
  const awaitingInput = useStore(scope.$awaitingInput)
  const activeQueueSessionKey = queueSessionKey || sessionId || null

  // Status items (subagents, background processes) are keyed by the RUNTIME
  // session id — gateway events and process.list both speak that id. Only the
  // queue uses the stored-session fallback key (prompts can queue pre-resume).
  const statusSessionId = sessionId ?? null

  // Coarse edge: re-renders ChatBar only when the stack shows/hides, NOT on
  // every per-item status mutation or other sessions' churn (see the hook).
  const statusPresent = useSessionStatusPresence(statusSessionId)

  // Publishes contributed micro actions for this session; the status stack
  // renders them as the pill strip at the top of the overlay lane.
  useComposerMicroActions(statusSessionId, busy)

  const composerRef = useRef<HTMLFormElement | null>(null)
  // The dock wraps the strips + status stack + composer; the thread's bottom
  // clearance measures this, while the pop-out drag still tracks the composer.
  const composerDockRef = useRef<HTMLDivElement | null>(null)
  const composerSurfaceRef = useRef<HTMLDivElement | null>(null)

  // Pop-out engine: docked↔floating state, dock/float/toggle, drag gestures, and
  // the on-screen re-clamp. Secondary windows can't pop out.
  const {
    dockProximity,
    dragging,
    handleComposerToggle,
    onComposerGesturePointerDown,
    popoutAllowed,
    popoutPosition,
    poppedOut
  } = useComposerPopout({ composerRef })

  // Coordinator-owned: the draft engine reads the live queue-edit snapshot off
  // this ref (to suppress its stash while editing a queued prompt) and the queue
  // engine writes it — an explicit shared handle, not a back-reference.
  const queueEditRef = useRef<QueueEditState | null>(null)
  const composingRef = useRef(false) // true during IME composition (CJK input)

  const { availableThemes, themeName } = useTheme()
  const at = useAtCompletions({ gateway: gateway ?? null, sessionId: sessionId ?? null, cwd: cwd ?? null })
  const slash = useSlashCompletions({ activeSkin: themeName, gateway: gateway ?? null, skinThemes: availableThemes })
  const emoji = useEmojiCompletions()

  const { t } = useI18n()
  const gatewayState = useStore($gatewayState)
  const reconnecting = gatewayState === 'closed' || gatewayState === 'error'
  const inputDisabled = disabled && !reconnecting

  // The draft engine — detached source of truth (DOM + draftRef + edge
  // selectors); typing never re-renders the chrome. ChatBar owns `queueEditRef`
  // and threads it in so the draft↔queue coupling is an explicit dep, not a tangle.
  const {
    activeQueueSessionKeyRef,
    clearDraft,
    draftRef,
    editorRef,
    focusInput,
    hasText,
    insertInlineRefs,
    insertText,
    isHelpHint,
    isSteerableText,
    loadIntoComposer,
    requestMainFocus,
    sessionIdRef,
    setComposerText,
    stashAt,
    syncDraftFromEditor
  } = useComposerDraft({ activeQueueSessionKey, focusKey, inputDisabled, queueEditRef, sessionId })

  // Undo/redo. The rich editor bypasses Chromium's editing pipeline for speed,
  // which also bypasses its undo stack — so we own the stack and every edit
  // path below banks its pre-edit state through `recordUndoPoint`.
  const { recordUndoPoint, redo, resetUndoHistory, undo, withUndoPoint } = useComposerUndo({
    editorRef,
    syncDraftFromEditor
  })

  // Prior history belongs to the draft that just left — undoing into another
  // conversation's text is worse than having none.
  useEffect(() => {
    resetUndoHistory()
  }, [activeQueueSessionKey, resetUndoHistory])

  // "Add URL" dialog — open/value state, autofocus, and submit (host onAddUrl or
  // an @url: directive into the draft).
  const { openUrlDialog, setUrlOpen, setUrlValue, submitUrl, urlInputRef, urlOpen, urlValue } = useComposerUrlDialog({
    insertText,
    onAddUrl
  })

  // The queue engine — queued turns, in-place editing, the shared drain lock,
  // and bounded auto-drain. Consumes the draft API and writes `queueEditRef`.
  const {
    beginQueuedEdit,
    drainNextQueued,
    editingQueuedPrompt,
    exitQueuedEdit,
    queueCurrentDraft,
    queueEdit,
    queueParked,
    queuedPrompts,
    sendQueuedNow,
    stepQueuedEdit
  } = useComposerQueue({
    activeQueueSessionKey,
    attachments,
    busy,
    clearDraft,
    draftRef,
    focusInput,
    loadIntoComposer,
    onCancel,
    onSubmit,
    queueEditRef,
    queueSessionKey,
    sessionId
  })

  const statusStackVisible = queuedPrompts.length > 0 || statusPresent

  // Halt vs. reach-the-queue: every interrupt lands on onCancel, but only the
  // gestures that MEAN "stop working" (Stop button, Esc) go through this
  // wrapper, which parks the queue first — an explicit halt must not roll
  // straight into the next queued prompt (that read as Stop not working; the
  // queued text also seemed to vanish, since the collapsed panel row was its
  // only trace). Interrupts that exist to advance the queue (send-now-while-
  // busy) call the raw onCancel and keep draining on settle. Parked entries
  // stay in the panel until resumed, sent, edited, or deleted.
  const haltRun = useCallback(() => {
    parkQueuedPrompts(activeQueueSessionKeyRef.current)

    return onCancel()
  }, [activeQueueSessionKeyRef, onCancel])

  const { compactPill, stacked } = useComposerMetrics({
    composerDockRef,
    composerRef,
    composerSurfaceRef,
    editorRef,
    poppedOut
  })

  const hasComposerPayload = hasText || attachments.length > 0
  const canSubmit = busy || hasComposerPayload

  // Steer only makes sense mid-turn, text-only (the gateway can't carry images
  // into a tool result) and never for a slash command (those execute inline).
  const canSteer = busy && !compacting && !!onSteer && attachments.length === 0 && isSteerableText

  // While busy: text redirects the live turn (Cursor-style stop-and-correct),
  // attachments queue for the next turn, an empty composer stops.
  const busyAction: 'steer' | 'queue' | 'stop' = canSteer
    ? 'steer'
    : compacting || hasComposerPayload
      ? 'queue'
      : 'stop'

  // The submit engine — the orchestration seam where draft + queue meet. Owns
  // the submit decision tree, the send-with-restore primitive, and steer.
  const { queueDraft, steerDraft, submitDraft } = useComposerSubmit({
    activeQueueSessionKey,
    activeQueueSessionKeyRef,
    attachments,
    busy,
    compacting,
    clearDraft,
    disabled,
    draftRef,
    drainNextQueued,
    editorRef,
    exitQueuedEdit,
    focusInput,
    inputDisabled,
    loadIntoComposer,
    // The submit engine's only cancel call is the Stop-button branch (busy +
    // empty composer) — an explicit halt, so it parks the queue.
    onCancel: haltRun,
    onSteer,
    onSubmit,
    queueCurrentDraft,
    queueEdit,
    queuedPrompts,
    sessionId,
    setComposerText,
    stashAt
  })

  // Resting / reconnecting / starting placeholder text, re-rolled only on a real
  // conversation change.
  const placeholder = useComposerPlaceholder({ disabled, reconnecting, sessionId })

  // Trigger / completion engine: @// detection, the adapter-driven item list,
  // popover selection, and chip insertion. The keydown nav block below consumes
  // this API; keyup uses triggerKeyConsumedRef to skip its refresh.
  const {
    argStageEmpty,
    ascendTriggerPath,
    closeTrigger,
    commitTypedSlashDirective,
    moveTriggerActive,
    refreshTrigger,
    replaceTriggerWithChip,
    setTriggerActive,
    slashFreeTextArgStage,
    trigger,
    triggerActive,
    triggerActiveExplicit,
    triggerItems,
    triggerKeyConsumedRef,
    triggerLoading
  } = useComposerTrigger({ at, draftRef, editorRef, emoji, recordUndoPoint, requestMainFocus, setComposerText, slash })

  // Pull the live contentEditable text into draftRef + the AUI composer state
  // (which drives `hasComposerPayload` → the send button). Shared by the input
  // and compositionend paths so committed IME text reaches state through either.
  // A pending coalesced flush (rAF id). `composerPlainText` serializes the whole
  // editor (O(n)), so running it on every event during a burst — holding a key,
  // or holding Cmd+V into a growing editor — is O(n²) across the burst. The
  // contentEditable DOM is the source of truth (submit + the compositionend /
  // keydown paths re-read it synchronously), so collapsing the input/paste
  // flushes to one per paint is lossless.
  const flushRafRef = useRef<number | undefined>(undefined)

  const flushEditorToDraft = (editor: HTMLDivElement) => {
    if (flushRafRef.current !== undefined) {
      window.cancelAnimationFrame(flushRafRef.current)
      flushRafRef.current = undefined
    }

    normalizeComposerEditorDom(editor)

    const nextDraft = sanitizeComposerInput(composerPlainText(editor))

    if (nextDraft !== draftRef.current) {
      draftRef.current = nextDraft
      setComposerText(nextDraft)
    }

    window.setTimeout(refreshTrigger, 0)
  }

  // Coalesce the high-frequency input/paste flushes to one per frame. Immediate
  // paths (compositionend, Enter/keydown, submit) keep calling
  // flushEditorToDraft directly, which cancels any pending coalesced run first.
  const scheduleFlushEditorToDraft = (editor: HTMLDivElement) => {
    if (flushRafRef.current !== undefined) {
      return
    }

    flushRafRef.current = window.requestAnimationFrame(() => {
      flushRafRef.current = undefined
      flushEditorToDraft(editor)
    })
  }

  useEffect(
    () => () => {
      if (flushRafRef.current !== undefined) {
        window.cancelAnimationFrame(flushRafRef.current)
      }
    },
    []
  )

  const handleEditorInput = (event: FormEvent<HTMLDivElement>) => {
    // During IME composition the DOM contains uncommitted preedit text
    // mixed with real content.  Skip state writes — compositionend flushes
    // the finalized text (see onCompositionEnd).
    if (composingRef.current) {
      return
    }

    scheduleFlushEditorToDraft(event.currentTarget)
  }

  // Native typing/deleting mutates the DOM through Chromium's editing pipeline,
  // whose undo stack we've taken over — so bank the pre-edit state here, before
  // the change lands. `beforeinput` is the only hook that still sees the old
  // text. Consecutive keystrokes coalesce into one entry, so ⌘Z steps back by a
  // burst rather than a character.
  const handleEditorBeforeInput = (event: FormEvent<HTMLDivElement>) => {
    const inputType = (event.nativeEvent as InputEvent).inputType

    // Undo/redo are ours (handled in useComposerUndo + keydown), and IME preedit
    // is not a committed edit — compositionend is where that text becomes real.
    if (inputType === 'historyUndo' || inputType === 'historyRedo' || composingRef.current) {
      return
    }

    recordUndoPoint({ coalesce: inputType === 'insertText' || inputType === 'deleteContentBackward' })
  }

  const handlePaste = (event: ClipboardEvent<HTMLDivElement>) => {
    const imageBlobs = extractClipboardImageBlobs(event.clipboardData)

    if (imageBlobs.length > 0) {
      event.preventDefault()

      if (onAttachImageBlob) {
        triggerHaptic('selection')

        for (const blob of imageBlobs) {
          void onAttachImageBlob(blob)
        }
      }

      return
    }

    // Trim surrounding whitespace so a copy that dragged along leading/trailing
    // blank lines (common when selecting from terminals, code blocks, web pages)
    // doesn't dump multiline padding into the composer. Internal newlines are
    // preserved — only the edges are cleaned up.
    const pastedText = sanitizeComposerInput(event.clipboardData.getData('text').trim())

    if (!pastedText) {
      event.preventDefault()

      // Under WSL2/WSLg the Windows host clipboard doesn't bridge *images* to
      // the Linux clipboard the DOM paste event reads, so a host screenshot
      // arrives as an empty paste (no blobs, no text). Fall back to the main
      // process, which pulls the image straight off the Windows clipboard.
      // Silent so a genuinely-empty paste doesn't pop a "no image" warning.
      if (onPasteClipboardImage) {
        triggerHaptic('selection')
        void onPasteClipboardImage({ silent: true })
      }

      return
    }

    if (DATA_IMAGE_URL_RE.test(pastedText)) {
      event.preventDefault()

      return
    }

    event.preventDefault()

    // Links in the paste land as `@url:` chips rather than a wall of URL text —
    // the same reference the "Add URL" dialog inserts, parsed in place so a link
    // mid-sentence keeps its position. Bare `@path` tokens promote the same way.
    // A paste into an open `@url:`/`@file:` scope CONSUMES that scope instead of
    // stacking on it — the scope is the browse mode the user is pasting into,
    // not text they typed and want to keep (`@url:@url:\`https://…\``).
    const scope = openDirectiveScope(event.currentTarget)

    recordUndoPoint()
    insertComposerContentsAtCaret(event.currentTarget, pathifyRefs(linkifyUrls(pastedText)), scope)
    scheduleFlushEditorToDraft(event.currentTarget)
  }

  const handleEditorKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    // IME composition: Enter confirms composed text, not a message submission.
    // We check both composingRef (set by compositionstart/compositionend, robust
    // across browsers) and nativeEvent.isComposing (Chromium fallback).  Without
    // this guard, pressing Enter to finalise a Korean/Japanese/Chinese IME
    // preedit fires submitDraft() and splits the message mid-word.
    if (composingRef.current || event.nativeEvent.isComposing) {
      return
    }

    // Undo/redo before anything else — we own the stack (see useComposerUndo),
    // so these never reach Chromium's native history, which has no record of
    // the Range-based edits the rich editor makes.
    if (isUndoShortcut(event.nativeEvent)) {
      event.preventDefault()
      undo()

      return
    }

    if (isRedoShortcut(event.nativeEvent)) {
      event.preventDefault()
      redo()

      return
    }

    // Plain Backspace right after a directive chip: remove the chip + its
    // auto-inserted trailing space as one unit, so deleting a directive never
    // leaves an orphaned space. (Modified backspaces stay native.)
    if (
      event.key === 'Backspace' &&
      !event.metaKey &&
      !event.ctrlKey &&
      !event.altKey &&
      withUndoPoint(() => deleteChipBeforeCaret(event.currentTarget))
    ) {
      event.preventDefault()
      flushEditorToDraft(event.currentTarget)

      return
    }

    // Non-collapsed Backspace/Delete: native selection-delete is ~O(n²) on large
    // drafts (Ctrl+A → Delete froze ~1.3s). Collapsed carets fall through.
    if (
      (event.key === 'Backspace' || event.key === 'Delete') &&
      withUndoPoint(() => deleteSelectionInEditor(event.currentTarget))
    ) {
      event.preventDefault()
      flushEditorToDraft(event.currentTarget)

      return
    }

    // A typed link finished with a space chips like a pasted one — the space
    // itself rides along inside the insert.
    if (withUndoPoint(() => chipTypedUrlOnSpace(event))) {
      event.preventDefault()
      flushEditorToDraft(event.currentTarget)

      return
    }

    // Same for a bare `@path` — a hand-typed or Tab-descended path chips into
    // the `@file:`/`@folder:` ref it means, instead of submitting as plain text
    // the backend never resolves.
    if (withUndoPoint(() => chipTypedPathOnSpace(event))) {
      event.preventDefault()
      flushEditorToDraft(event.currentTarget)

      return
    }

    // Cmd/Ctrl+Shift+K drains the next queued message. Plain Cmd/Ctrl+K is
    // reserved for the global command palette.
    if ((event.metaKey || event.ctrlKey) && !event.altKey && event.shiftKey && event.key.toLowerCase() === 'k') {
      event.preventDefault()

      if (!busy) {
        void drainNextQueued()
      }

      return
    }

    // The popover is open but its items are still in flight (debounce + RPC).
    // Tab must not fall through to the browser — it would move focus out of
    // the composer mid-completion, which reads as the popover "eating" the
    // keypress. Swallow it; the refresh lands with the items.
    if (trigger && triggerLoading && triggerItems.length === 0 && event.key === 'Tab') {
      event.preventDefault()
      triggerKeyConsumedRef.current = true

      return
    }

    if (trigger && triggerItems.length > 0) {
      if (event.key === 'ArrowDown') {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        moveTriggerActive(1)

        return
      }

      if (event.key === 'ArrowUp') {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        moveTriggerActive(-1)

        return
      }

      // Accepting the highlighted item: a no-arg command commits its directive
      // chip, an arg-taking command expands to its options step, and an arg
      // option commits the full `/cmd arg` chip.
      const accept = acceptsTriggerCompletion({
        activeExplicit: triggerActiveExplicit,
        freeTextArgStage: slashFreeTextArgStage,
        key: event.key,
        kind: trigger.kind,
        query: trigger.query
      })

      if (accept) {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        const item = triggerItems[triggerActive]

        if (item) {
          // Tab means "go deeper" on a folder; Enter means "I want this one".
          // Everything else treats them alike.
          replaceTriggerWithChip(item, { descend: event.key === 'Tab' })
        }

        return
      }

      // Backspace climbs out of an `@` path one segment at a time, mirroring
      // Tab's one-key descent. Only when the caret sits at the end of the
      // token — mid-token editing keeps normal character deletion.
      if (event.key === 'Backspace' && !event.metaKey && !event.altKey && ascendTriggerPath()) {
        event.preventDefault()
        triggerKeyConsumedRef.current = true

        return
      }

      if (event.key === 'Escape') {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        closeTrigger()

        return
      }
    }

    // Arg stage with nothing left to suggest — a fully-typed arg the backend
    // completer no longer echoes (it drops the exact match), e.g.
    // `/personality creative`. Space/Tab still commit what's typed as a single
    // directive chip; Enter falls through to submit (send it as-is).
    if (
      trigger?.kind === '/' &&
      !triggerItems.length &&
      (event.key === ' ' || event.key === 'Tab') &&
      slashArgStage(trigger.query) &&
      trigger.query.trim()
    ) {
      if (commitTypedSlashDirective()) {
        event.preventDefault()
        triggerKeyConsumedRef.current = true

        return
      }
    }

    // ArrowUp/ArrowDown navigate, in priority order: the queue (edit entries in
    // place) then sent-message history. The history ring is derived from live
    // session messages each press — single source of truth, no mirror.
    if (event.key === 'ArrowUp') {
      const currentDraft = draftRef.current

      // Editing a queued turn → walk to the older entry.
      if (queueEdit && stepQueuedEdit(-1)) {
        event.preventDefault()
        triggerKeyConsumedRef.current = true

        return
      }

      // Empty composer + a queued turn → open the newest queued entry for edit
      // (the row's pencil), not a text recall. Enter saves it back to the queue.
      if (!currentDraft.trim() && !queueEdit && queuedPrompts.length > 0) {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        beginQueuedEdit(queuedPrompts[queuedPrompts.length - 1]!)

        return
      }

      // Don't hijack a typed draft unless already browsing — they'd lose it.
      if (currentDraft.trim() && !isBrowsingHistory(sessionId)) {
        return
      }

      event.preventDefault()
      triggerKeyConsumedRef.current = true

      // $messages is read imperatively (not subscribed) so the composer
      // doesn't re-render on every streaming delta flush.
      const history = deriveUserHistory(scope.$messages.get(), chatMessageText)
      const entry = browseBackward(sessionId, currentDraft, history)

      if (entry !== null) {
        loadIntoComposer(entry, scope.attachments.$attachments.get())
      }

      return
    }

    if (event.key === 'ArrowDown') {
      // Editing a queued turn → walk to the newer entry (past the newest exits).
      if (queueEdit) {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        stepQueuedEdit(1)

        return
      }

      // Browsing sent history → step toward the present, restoring the draft.
      if (isBrowsingHistory(sessionId)) {
        event.preventDefault()
        triggerKeyConsumedRef.current = true

        const history = deriveUserHistory(scope.$messages.get(), chatMessageText)
        const result = browseForward(sessionId, history)

        if (result !== null) {
          loadIntoComposer(result.text, scope.attachments.$attachments.get())
        }
      }

      return
    }

    // Cmd/Ctrl+Enter queues a follow-up while a turn runs. Plain Enter steers
    // a text-only draft, so both live-turn actions stay reachable by keyboard.
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && !event.shiftKey) {
      event.preventDefault()

      if (busy && !disabled) {
        // As with plain Enter, source the just-typed content from the DOM so a
        // fast keypress cannot queue a stale draft.
        const editorText = editorRef.current ? composerPlainText(editorRef.current) : draftRef.current

        if (editorText !== draftRef.current) {
          draftRef.current = editorText
          setComposerText(editorText)
        }

        queueDraft()
      }

      return
    }

    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()

      // Decide from the DOM, not React state. `hasComposerPayload` is derived
      // from the AUI composer state, which lags the latest keystroke by a
      // render, so on fast typing / IME the just-typed text isn't in state yet.
      // Without the live read, a real message typed while prompts are queued
      // would drain the queue instead of sending. submitDraft() re-syncs and
      // sends the live editor text.
      const editorText = editorRef.current ? composerPlainText(editorRef.current) : draftRef.current
      const hasLivePayload = editorText.trim().length > 0 || attachments.length > 0

      if (disabled) {
        return
      }

      if (!busy && !hasLivePayload && queuedPrompts.length > 0) {
        void drainNextQueued()

        return
      }

      // Empty Enter while busy. With prompts queued this is the double-send:
      // the first Enter put the words in the queue, a second sends them now
      // (promote + interrupt + drain on settle), mirroring the idle empty-Enter
      // drain above. With nothing queued it stays a no-op — interrupting is
      // explicit (Stop/Esc), never a stray Enter after sending. Gate on the live
      // DOM payload (not the render-lagged composer state) so a message typed
      // fast / via IME while busy still reaches submitDraft() and gets queued
      // instead of being mistaken for an empty Enter.
      if (busy && !hasLivePayload) {
        const head = queuedPrompts.find(entry => entry.id !== queueEdit?.entryId)

        if (head) {
          sendQueuedNow(head.id)
        }

        return
      }

      submitDraft()

      return
    }

    if (event.key === 'Escape') {
      // Editing a queued turn → Esc cancels the edit, restoring the prior draft.
      if (queueEdit) {
        event.preventDefault()
        exitQueuedEdit('cancel')

        return
      }

      // Otherwise Esc interrupts the running turn (Stop-button parity) — unless
      // the turn is parked waiting on the user, where Esc must not discard the
      // pending prompt. An explicit halt, so it parks the queue too.
      if (busy && !awaitingInput) {
        event.preventDefault()
        triggerHaptic('cancel')
        void Promise.resolve(haltRun())
      }
    }
  }

  const handleEditorKeyUp = () => {
    // If this keyup belongs to a key the open trigger popover already consumed
    // in keydown (Arrow/Enter/Tab/Escape), skip the refresh. Those keys never
    // edit text, and for Escape the keydown already closed the menu — a refresh
    // here would re-detect the still-present `/` and instantly reopen it. We
    // read a ref set during keydown rather than `trigger`, because by keyup
    // time React has re-rendered and `trigger` may already be null.
    if (triggerKeyConsumedRef.current) {
      triggerKeyConsumedRef.current = false

      return
    }

    window.setTimeout(refreshTrigger, 0)
  }

  const {
    dragActive,
    handleDragEnter,
    handleDragLeave,
    handleDragOver,
    handleDrop,
    handleInputDragOver,
    handleInputDrop
  } = useComposerDrop({ cwd, insertInlineRefs, onAttachDroppedItems, requestMainFocus })

  // Branch / worktree hand-offs (CodingStatusRow). Owns the worktree open +
  // branch-off/convert/list/switch actions; draft travels into the new session.
  const { handleBranchOff, handleConvertBranch, handleListBranches, handleSwitchBranch, openInWorktree } =
    useComposerBranch({ clearDraft, cwd, draftRef })

  // Global Esc-to-cancel when the chat (not the composer input) has focus.
  // Same explicit-halt semantics as the Stop button: park the queue.
  useComposerEscCancel({ awaitingInput, busy, onCancel: haltRun, target: scope.target })

  const {
    conversation,
    dictate,
    endConversation,
    handleToggleAutoSpeak,
    startConversation,
    voiceActivityState,
    voiceConversationActive,
    voiceStatus
  } = useComposerVoice({
    busy,
    clearDraft,
    disabled,
    focusInput,
    insertText,
    maxRecordingSeconds,
    // Voice barge-in mid-generation halts the run like the Stop button.
    onInterrupt: haltRun,
    onSubmit,
    onTranscribeAudio,
    sessionId,
    target: scope.target
  })

  // Keep the typed-stop interceptor (see onSubmit above) in sync with the
  // live conversation state. Render-time ref assignment, same pattern as
  // dispatchSubmitRef — no effect needed for a plain mirror.
  voiceStopRef.current = { active: voiceConversationActive, end: endConversation }

  const contextMenu = (
    <ContextMenu
      onInsertText={insertText}
      onOpenUrlDialog={openUrlDialog}
      onPasteClipboardImage={onPasteClipboardImage}
      onPickFiles={onPickFiles}
      onPickFolders={onPickFolders}
      onPickImages={onPickImages}
      state={state}
    />
  )

  const controls = (
    <ComposerControls
      autoSpeak={autoSpeak}
      busy={busy}
      busyAction={busyAction}
      canSubmit={canSubmit}
      compactModelPill={poppedOut || compactPill}
      conversation={{
        active: voiceConversationActive,
        level: conversation.level,
        muted: conversation.muted,
        onEnd: endConversation,
        onStart: startConversation,
        onStopTurn: conversation.stopTurn,
        onToggleMute: conversation.toggleMute,
        status: conversation.status
      }}
      disabled={disabled}
      hasComposerPayload={hasComposerPayload}
      onDictate={dictate}
      onQueue={queueDraft}
      onToggleAutoSpeak={handleToggleAutoSpeak}
      state={state}
      voiceStatus={voiceStatus}
    />
  )

  const input = (
    <div className={cn('relative', stacked ? 'w-full' : 'min-w-(--composer-input-inline-min-width) flex-1')}>
      <div
        aria-disabled={inputDisabled ? true : undefined}
        aria-label={t.composer.message}
        autoCapitalize="off"
        autoCorrect="off"
        className={cn(
          'min-h-[1.625rem] min-h-(--composer-input-min-height) max-h-(--composer-input-max-height) cursor-text overflow-y-auto whitespace-pre-wrap break-words [overflow-wrap:anywhere] bg-transparent pb-1 pr-1 pt-1 leading-normal text-foreground outline-none disabled:cursor-not-allowed',
          COMPOSER_PLACEHOLDER_CLASS,
          '**:data-ref-text:cursor-default',
          stacked && 'pl-3',
          stacked ? 'w-full' : 'min-w-(--composer-input-inline-min-width) flex-1'
        )}
        contentEditable={!inputDisabled}
        data-placeholder={placeholder}
        data-slot={RICH_INPUT_SLOT}
        onBeforeInput={handleEditorBeforeInput}
        onBlur={() => window.setTimeout(closeTrigger, 80)}
        onCompositionEnd={event => {
          composingRef.current = false

          // The input events fired *during* composition were skipped (they
          // carried uncommitted preedit text), and Chromium does NOT reliably
          // emit a trailing input event after compositionend on Windows IMEs.
          // Without flushing here, committed multi-character IME input (e.g.
          // Chinese "你好", Japanese, Korean) never reaches composer state, so
          // `hasComposerPayload` stays false and the send button stays hidden
          // until an unrelated edit forces a sync (#39614).
          flushEditorToDraft(event.currentTarget)
        }}
        onCompositionStart={() => {
          composingRef.current = true
        }}
        onDragOver={handleInputDragOver}
        onDrop={handleInputDrop}
        onFocus={() => markActiveComposer(scope.target)}
        onInput={handleEditorInput}
        onKeyDown={handleEditorKeyDown}
        onKeyUp={handleEditorKeyUp}
        onMouseUp={refreshTrigger}
        onPaste={handlePaste}
        ref={editorRef}
        role="textbox"
        spellCheck={false}
        suppressContentEditableWarning
      />
      {/* assistant-ui requires ComposerPrimitive.Input somewhere in the tree
        so the composer-state binding (text + IME + paste + form-submit hookup)
        wires up. We render the real input UI ourselves above via the
        contentEditable, so the primitive is invisible (sr-only).

        IMPORTANT: don't let it render its default <TextareaAutosize>. That
        component runs `useLayoutEffect(resizeTextarea)` on every value change
        and reads `node.scrollHeight` against a hidden measurement textarea,
        forcing two synchronous layouts per keystroke for an element the
        user can't see. Profiling 400-char synthetic typing showed >900ms
        cumulative cost in getHeight2/calculateNodeHeight alone (~2.3ms/key)
        on top of the per-keystroke React commit.

        `asChild` swaps TextareaAutosize for a Radix Slot wrapping our
        plain <textarea>, which carries the binding but skips autosize. */}
      <ComposerPrimitive.Input asChild submitMode="ctrlEnter" tabIndex={-1} unstable_focusOnScrollToBottom={false}>
        <textarea
          aria-hidden
          autoCapitalize="off"
          autoComplete="off"
          autoCorrect="off"
          className="sr-only"
          spellCheck={false}
          tabIndex={-1}
        />
      </ComposerPrimitive.Input>
    </div>
  )

  return (
    <>
      {dragging && poppedOut && (
        <div
          aria-hidden
          // `absolute`, not `fixed`: anchor to the chat-column root (the same
          // `relative isolate` container the docked composer centers in) so the
          // glow spans the thread area only — never the full viewport / under the
          // sidebar. The dock target IS the docked position, so they must share
          // a containing block.
          className="pointer-events-none absolute inset-x-0 bottom-0 z-20 h-32"
          style={{
            // A bottom-centered radial glow — soft on every side by construction,
            // so it reads as the dock target without any hard band edges. Its
            // intensity tracks how close the composer is to the dock (1 = peak).
            background:
              'radial-gradient(64% 130% at 50% 100%, color-mix(in srgb, var(--color-primary) 26%, transparent) 0%, transparent 70%)',
            // Scaled by --dock-glow-scale (lower in light mode — see styles.css).
            opacity: `calc(${0.1 + dockProximity * 0.57} * var(--dock-glow-scale, 1))`
          }}
        />
      )}
      <ComposerPrimitive.Unstable_TriggerPopoverRoot>
        {/* Dock column: owns the composer's POSITION and stacks, bottom-up,
            [micro actions] · [status stack] · [composer] · [underside].
            Anchored at the bottom, so in-flow children grow upward and still
            overlay the thread — no absolute lane needed.

            The strips are siblings of the composer, not children: the pop-out
            drag region is `absolute inset-0` INSIDE the composer, so anything
            rendered in there is inside the grab area by construction. Keeping
            them out here is what makes that impossible rather than excluded. */}
        <div
          className={cn(
            'z-30 flex flex-col',
            poppedOut ? 'fixed max-w-[calc(100vw-1.5rem)]' : 'absolute bottom-0 left-1/2 max-w-full -translate-x-1/2'
          )}
          data-popped-out={poppedOut ? '' : undefined}
          data-slot="composer-dock"
          data-thread-scrolled-up={scrolledUp ? '' : undefined}
          // Measured for the thread's bottom clearance: the dock is the box
          // that contains the strips, the status stack, AND the composer, so
          // one measurement covers everything the thread must clear.
          ref={composerDockRef}
          style={
            poppedOut
              ? {
                  bottom: `${popoutPosition.bottom}px`,
                  right: `${popoutPosition.right}px`,
                  // A compact one-sentence width when floating.
                  ['--composer-popout-width' as string]: `${POPOUT_WIDTH_REM}rem`
                }
              : undefined
          }
        >
          {/* Aligned to the composer SURFACE, which sits inside the composer's
              5px transparent grab margin — so both strips carry the same inset
              and share one left edge with it. */}
          <div className={cn(composerFloatingStrip, 'px-[5px] pb-1.5 empty:hidden')}>
            <ActionBadges sessionId={statusSessionId} />
          </div>
          {/* Session-scoped status stack (todos, subagents, background tasks,
              queue). An in-flow dock child: the dock is bottom-anchored, so it
              grows upward over the thread and the dock's own measurement covers
              it. Collapses to nothing when every status is empty. */}
          <ComposerStatusStack
            queue={
              activeQueueSessionKey && queuedPrompts.length > 0 ? (
                <QueuePanel
                  busy={busy}
                  editingId={queueEdit?.entryId ?? null}
                  entries={queuedPrompts}
                  onDelete={id => {
                    if (removeQueuedPrompt(activeQueueSessionKey, id) && queueEdit?.entryId === id) {
                      exitQueuedEdit('cancel')
                    }
                  }}
                  onEdit={beginQueuedEdit}
                  onResume={() => {
                    unparkQueuedPrompts(activeQueueSessionKey)

                    // Idle → kick the head immediately; busy → the settle drain
                    // takes over now that the park is lifted.
                    if (!busy) {
                      void drainNextQueued()
                    }
                  }}
                  onSendNow={id => void sendQueuedNow(id)}
                  parked={queueParked}
                />
              ) : null
            }
            sessionId={statusSessionId}
          />
          <ComposerPrimitive.Root
            className={cn(
              'group/composer relative w-full overflow-visible rounded-2xl',
              poppedOut && 'bg-transparent',
              dragging && 'cursor-grabbing select-none touch-none'
            )}
            data-drag-active={dragActive ? '' : undefined}
            data-popped-out={poppedOut ? '' : undefined}
            data-slot="composer-root"
            data-status-stack={statusStackVisible ? '' : undefined}
            data-thread-scrolled-up={scrolledUp ? '' : undefined}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onPointerDown={popoutAllowed ? onComposerGesturePointerDown : undefined}
            onSubmit={e => {
              e.preventDefault()

              if (composingRef.current) {
                return
              }

              submitDraft()
            }}
            ref={composerRef}
          >
            {isHelpHint && <HelpHint />}
            {trigger && !argStageEmpty && (
              <ComposerTriggerPopover
                activeIndex={triggerActive}
                items={triggerItems}
                kind={trigger.kind}
                loading={triggerLoading}
                onHover={setTriggerActive}
                onPick={replaceTriggerWithChip}
                scope={trigger.scope}
              />
            )}
            {!poppedOut && (
              <div
                className="pointer-events-none absolute inset-0 rounded-[inherit]"
                style={{ background: COMPOSER_FADE_BACKGROUND }}
              />
            )}
            {/* Drag region: covers the transparent grab margin around the surface.
              The surface sits on top (z-4) so only the exposed ring receives this
              element's hover/cursor — grab cursor + a diagonal hatch (/////)
              appear when you hover the draggable margin, never over the input.
              The hatch pattern + opacity ladder live in styles.css. */}
            {popoutAllowed && (
              <div
                aria-hidden
                className={cn('pointer-events-auto absolute inset-0', dragging ? 'cursor-grabbing' : 'cursor-grab')}
                data-dragging={dragging ? '' : undefined}
                data-slot="composer-drag-region"
                onDoubleClick={handleComposerToggle}
              />
            )}
            <div className="relative w-full rounded-[inherit]">
              <div
                className={cn(
                  'group/composer-surface relative z-4 isolate grid grid-rows-[auto_1fr] overflow-hidden rounded-[inherit] border border-[color-mix(in_srgb,var(--dt-composer-ring)_calc(18%*var(--composer-ring-strength)),var(--dt-input))]',
                  COMPOSER_DROP_FADE_CLASS,
                  dragActive && COMPOSER_DROP_ACTIVE_CLASS
                )}
                data-slot="composer-surface"
                ref={composerSurfaceRef}
              >
                <div
                  aria-hidden
                  className={cn(
                    'pointer-events-none absolute inset-0 -z-10 rounded-[inherit]',
                    composerFill,
                    composerSurfaceGlass
                  )}
                />
                <CodingStatusRow
                  onBranchOff={handleBranchOff}
                  onConvertBranch={handleConvertBranch}
                  onListBranches={handleListBranches}
                  // A tile's rail reviews ITS worktree: pin the pane's scope to
                  // this surface's cwd. Main keeps the classic follow-the-
                  // active-session scope (null).
                  onOpen={() => toggleReview(scope.target === 'main' ? null : (cwd ?? null))}
                  onOpenWorktree={openInWorktree}
                  onSwitchBranch={handleSwitchBranch}
                  repoPath={cwd}
                />
                <div
                  className={cn(
                    'relative z-1 flex min-h-0 w-full flex-col gap-(--composer-row-gap) overflow-hidden rounded-[inherit] px-(--composer-surface-pad-x) py-(--composer-surface-pad-y) transition-opacity duration-200 ease-out',
                    scrolledUp
                      ? 'opacity-30 group-hover/composer:opacity-100 group-focus-within/composer-surface:opacity-100'
                      : 'opacity-100'
                  )}
                  data-slot="composer-fade"
                >
                  {/* Contribution seams: banners above, a row below, inline
                    additions beside the "+" menu and before the controls.
                    All four render nothing until something contributes. */}
                  <ContribSlot area={COMPOSER_AREAS.top} />
                  <VoiceActivity state={voiceActivityState} />
                  <VoicePlaybackActivity />
                  {queueEdit && editingQueuedPrompt && (
                    <div className="flex items-center justify-between gap-2 rounded-lg border border-[color-mix(in_srgb,var(--dt-composer-ring)_32%,transparent)] bg-accent/18 px-2 py-1">
                      <div className="min-w-0 text-[0.7rem] text-muted-foreground/88">
                        {t.composer.editingQueuedInComposer}
                      </div>
                      <div className="flex shrink-0 items-center gap-1">
                        <Button
                          className="h-6 rounded-md px-2 text-[0.68rem]"
                          onClick={() => exitQueuedEdit('cancel')}
                          type="button"
                          variant="ghost"
                        >
                          {t.common.cancel}
                        </Button>
                        <Button
                          className="h-6 rounded-md px-2 text-[0.68rem]"
                          onClick={() => exitQueuedEdit('save')}
                          type="button"
                        >
                          {t.common.save}
                        </Button>
                      </div>
                    </div>
                  )}
                  {attachments.length > 0 && <AttachmentList attachments={attachments} onRemove={onRemoveAttachment} />}
                  <div
                    className={cn(
                      'grid w-full',
                      stacked
                        ? 'grid-cols-[auto_1fr] gap-(--composer-row-gap) [grid-template-areas:"input_input"_"menu_controls"]'
                        : 'grid-cols-[auto_1fr_auto] items-center gap-(--composer-control-gap) [grid-template-areas:"menu_input_controls"]'
                    )}
                  >
                    <div className="flex translate-y-[3px] items-start gap-(--composer-control-gap) self-start [grid-area:menu]">
                      {contextMenu}
                      <ContribSlot area={COMPOSER_AREAS.leading} />
                    </div>
                    <div className="min-w-0 [grid-area:input]">{input}</div>
                    <div className="flex items-center justify-end gap-(--composer-control-gap) [grid-area:controls]">
                      <ContribSlot area={COMPOSER_AREAS.actions} />
                      {controls}
                    </div>
                  </div>
                  <ContribSlot area={COMPOSER_AREAS.bottom} />
                </div>
              </div>
            </div>
          </ComposerPrimitive.Root>
          {/* Underside: chrome-free strip BELOW the composer. Outside the root
              for the same reason as the micro actions — it must not fall inside
              the pop-out drag region. Same px as the strip above, so the two
              bracket the composer on one vertical line. */}
          <div className={cn(composerFloatingStrip, 'px-[5px] pt-1.5 empty:hidden')}>
            <ContribSlot area={COMPOSER_AREAS.underside} />
          </div>
        </div>
      </ComposerPrimitive.Unstable_TriggerPopoverRoot>

      <UrlDialog
        inputRef={urlInputRef}
        onChange={setUrlValue}
        onOpenChange={setUrlOpen}
        onSubmit={submitUrl}
        open={urlOpen}
        value={urlValue}
      />
    </>
  )
}

export function ChatBarFallback() {
  return (
    <div
      className={cn(
        'group/composer absolute bottom-0 left-1/2 z-30 w-[min(var(--composer-width),calc(100%-2rem))] max-w-full -translate-x-1/2 rounded-2xl pt-2 pb-[var(--composer-shell-pad-block-end)]',
        'bg-linear-to-b from-transparent to-background/55'
      )}
      data-slot="composer-root"
    >
      <div className="composer-fallback-surface relative isolate h-(--composer-fallback-height) w-full rounded-[inherit] border border-[color-mix(in_srgb,var(--dt-composer-ring)_calc(18%*var(--composer-ring-strength)),var(--dt-input))]">
        <div
          aria-hidden
          className={cn(
            'pointer-events-none absolute inset-0 -z-10 rounded-[inherit]',
            composerFill,
            composerSurfaceGlass
          )}
        />
      </div>
    </div>
  )
}
