import { ComposerPrimitive } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import {
  type ClipboardEvent,
  type FormEvent,
  type KeyboardEvent,
  useEffect,
  useRef
} from 'react'

import { composerFill, composerSurfaceGlass } from '@/components/chat/composer-dock'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { chatMessageText } from '@/lib/chat-messages'
import { DATA_IMAGE_URL_RE } from '@/lib/embedded-images'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'
import { $composerAttachments } from '@/store/composer'
import { browseBackward, browseForward, deriveUserHistory, isBrowsingHistory } from '@/store/composer-input-history'
import { POPOUT_WIDTH_REM } from '@/store/composer-popout'
import { removeQueuedPrompt } from '@/store/composer-queue'
import { $activeSessionAwaitingInput } from '@/store/prompts'
import { toggleReview } from '@/store/review'
import { $gatewayState, $messages } from '@/store/session'
import { $threadScrolledUp } from '@/store/thread-scroll'
import { $autoSpeakReplies } from '@/store/voice-prefs'
import { useTheme } from '@/themes'

import { AttachmentList } from './attachments'
import {
  COMPOSER_FADE_BACKGROUND,
  type QueueEditState,
  slashArgStage
} from './composer-utils'
import { ContextMenu } from './context-menu'
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
import { useComposerUrlDialog } from './hooks/use-composer-url-dialog'
import { useComposerVoice } from './hooks/use-composer-voice'
import { useSlashCompletions } from './hooks/use-slash-completions'
import { useSessionStatusPresence } from './hooks/use-status-presence'
import { QueuePanel } from './queue-panel'
import {
  composerPlainText,
  deleteChipBeforeCaret,
  deleteSelectionInEditor,
  insertPlainTextAtCaret,
  normalizeComposerEditorDom,
  RICH_INPUT_SLOT
} from './rich-editor'
import { ComposerStatusStack } from './status-stack'
import { CodingStatusRow } from './status-stack/coding-row'
import { extractClipboardImageBlobs } from './text-utils'
import { ComposerTriggerPopover } from './trigger-popover'
import type { ChatBarProps } from './types'
import { UrlDialog } from './url-dialog'
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
  onSubmit,
  onTranscribeAudio
}: ChatBarProps) {
  const attachments = useStore($composerAttachments)
  const scrolledUp = useStore($threadScrolledUp)
  const autoSpeak = useStore($autoSpeakReplies)
  // The turn is parked on the user (clarify / approval / sudo / secret). Esc must
  // not interrupt it — there's nothing actively running to stop, and stopping
  // would discard a question the user may want to come back to. The blocking
  // prompt owns its own dismissal (Skip, Reject, dialog close).
  const awaitingInput = useStore($activeSessionAwaitingInput)
  const activeQueueSessionKey = queueSessionKey || sessionId || null

  // Status items (subagents, background processes) are keyed by the RUNTIME
  // session id — gateway events and process.list both speak that id. Only the
  // queue uses the stored-session fallback key (prompts can queue pre-resume).
  const statusSessionId = sessionId ?? null

  // Coarse edge: re-renders ChatBar only when the stack shows/hides, NOT on
  // every per-item status mutation or other sessions' churn (see the hook).
  const statusPresent = useSessionStatusPresence(statusSessionId)

  const composerRef = useRef<HTMLFormElement | null>(null)
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
    stashAt
  } = useComposerDraft({ activeQueueSessionKey, focusKey, inputDisabled, queueEditRef, sessionId })

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

  const { stacked } = useComposerMetrics({ composerRef, composerSurfaceRef, editorRef, poppedOut })
  const hasComposerPayload = hasText || attachments.length > 0
  const canSubmit = busy || hasComposerPayload
  const busyAction = busy && hasComposerPayload ? 'queue' : 'stop'

  // Steer only makes sense mid-turn, text-only (the gateway can't carry images
  // into a tool result) and never for a slash command (those execute inline).
  const canSteer = busy && !!onSteer && attachments.length === 0 && isSteerableText

  const showHelpHint = isHelpHint

  // The submit engine — the orchestration seam where draft + queue meet. Owns
  // the submit decision tree, the send-with-restore primitive, and steer.
  const { steerDraft, submitDraft } = useComposerSubmit({
    activeQueueSessionKey,
    activeQueueSessionKeyRef,
    attachments,
    busy,
    canSteer,
    clearDraft,
    disabled,
    draftRef,
    drainNextQueued,
    editorRef,
    exitQueuedEdit,
    focusInput,
    inputDisabled,
    loadIntoComposer,
    onCancel,
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
    closeTrigger,
    commitTypedSlashDirective,
    refreshTrigger,
    replaceTriggerWithChip,
    setTriggerActive,
    trigger,
    triggerActive,
    triggerItems,
    triggerKeyConsumedRef,
    triggerLoading
  } = useComposerTrigger({ at, draftRef, editorRef, requestMainFocus, setComposerText, slash })

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

    const nextDraft = composerPlainText(editor)

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
    const pastedText = event.clipboardData.getData('text').trim()

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
    insertPlainTextAtCaret(event.currentTarget, pastedText)
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

    // Plain Backspace right after a directive chip: remove the chip + its
    // auto-inserted trailing space as one unit, so deleting a directive never
    // leaves an orphaned space. (Modified backspaces stay native.)
    if (
      event.key === 'Backspace' &&
      !event.metaKey &&
      !event.ctrlKey &&
      !event.altKey &&
      deleteChipBeforeCaret(event.currentTarget)
    ) {
      event.preventDefault()
      flushEditorToDraft(event.currentTarget)

      return
    }

    // Non-collapsed Backspace/Delete: native selection-delete is ~O(n²) on large
    // drafts (Ctrl+A → Delete froze ~1.3s). Collapsed carets fall through.
    if ((event.key === 'Backspace' || event.key === 'Delete') && deleteSelectionInEditor(event.currentTarget)) {
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

    if (trigger && triggerItems.length > 0) {
      if (event.key === 'ArrowDown') {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        setTriggerActive(idx => (idx + 1) % triggerItems.length)

        return
      }

      if (event.key === 'ArrowUp') {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        setTriggerActive(idx => (idx - 1 + triggerItems.length) % triggerItems.length)

        return
      }

      // Enter / Tab / Space all accept the highlighted item: a no-arg command
      // commits its directive chip, an arg-taking command expands to its
      // options step, and an arg option commits the full `/cmd arg` chip. Space
      // is slash-only (an `@` mention takes a literal space) and gated to a
      // non-empty query so a bare `/ ` still types a space.
      const acceptOnSpace = event.key === ' ' && trigger.kind === '/' && Boolean(trigger.query.trim())
      const accept = event.key === 'Enter' || event.key === 'Tab' || acceptOnSpace

      if (accept) {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        const item = triggerItems[triggerActive]

        if (item) {
          replaceTriggerWithChip(item)
        }

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
      event.preventDefault()
      triggerKeyConsumedRef.current = true
      commitTypedSlashDirective()

      return
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
      const history = deriveUserHistory($messages.get(), chatMessageText)
      const entry = browseBackward(sessionId, currentDraft, history)

      if (entry !== null) {
        loadIntoComposer(entry, $composerAttachments.get())
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

        const history = deriveUserHistory($messages.get(), chatMessageText)
        const result = browseForward(sessionId, history)

        if (result !== null) {
          loadIntoComposer(result.text, $composerAttachments.get())
        }
      }

      return
    }

    // Cmd/Ctrl+Enter is reserved for steering the live run — never a send.
    // Steer when there's a steerable draft, otherwise swallow it so it can't
    // surprise-send. (Plain Enter still queues while busy / sends when idle.)
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && !event.shiftKey) {
      event.preventDefault()

      if (canSteer) {
        steerDraft()
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

      // Empty Enter while busy is a no-op — interrupting is explicit (Stop/Esc),
      // never a stray Enter after sending. With a payload, submitDraft queues it.
      // Gate on the live DOM payload (not the render-lagged composer state) so a
      // message typed fast / via IME while busy still reaches submitDraft() and
      // gets queued instead of being mistaken for an empty Enter.
      if (busy && !hasLivePayload) {
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
      // pending prompt.
      if (busy && !awaitingInput) {
        event.preventDefault()
        triggerHaptic('cancel')
        void Promise.resolve(onCancel())
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
  useComposerEscCancel({ awaitingInput, busy, onCancel })

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
    onSubmit,
    onTranscribeAudio,
    sessionId
  })

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
      canSteer={canSteer}
      canSubmit={canSubmit}
      compactModelPill={poppedOut}
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
      onSteer={steerDraft}
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
          'min-h-(--composer-input-min-height) max-h-(--composer-input-max-height) cursor-text overflow-y-auto whitespace-pre-wrap break-words [overflow-wrap:anywhere] bg-transparent pb-1 pr-1 pt-1 leading-normal text-foreground outline-none disabled:cursor-not-allowed',
          'empty:before:content-[attr(data-placeholder)] empty:before:text-muted-foreground/60',
          '**:data-ref-text:cursor-default',
          stacked && 'pl-3',
          stacked ? 'w-full' : 'min-w-(--composer-input-inline-min-width) flex-1'
        )}
        contentEditable={!inputDisabled}
        data-placeholder={placeholder}
        data-slot={RICH_INPUT_SLOT}
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
        onFocus={() => markActiveComposer('main')}
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
        <ComposerPrimitive.Root
          className={cn(
            'group/composer z-30 overflow-visible rounded-2xl',
            poppedOut
              ? // Floating: the composer (with its own border) floats with an even
                // 5px transparent grab margin around it — drag that to move it.
                'fixed w-[var(--composer-popout-width)] max-w-[calc(100vw-1.5rem)] bg-transparent p-[5px]'
              : 'absolute bottom-0 left-1/2 w-[min(var(--composer-width),calc(100%-2rem))] max-w-full -translate-x-1/2 pt-2 pb-[var(--composer-shell-pad-block-end)]',
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
          {showHelpHint && <HelpHint />}
          {trigger && !argStageEmpty && (
            <ComposerTriggerPopover
              activeIndex={triggerActive}
              items={triggerItems}
              kind={trigger.kind}
              loading={triggerLoading}
              onHover={setTriggerActive}
              onPick={replaceTriggerWithChip}
            />
          )}
          {/* Session-scoped status stack (todos, subagents, background tasks,
              queue). Out of flow so it never inflates the composer's measured
              height; it overlays the chat instead of pushing it, and publishes
              its own --status-stack-measured-height so the thread's clearance
              accounts for it. Collapses to nothing when every status is empty. */}
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
                  onSendNow={id => void sendQueuedNow(id)}
                />
              ) : null
            }
            sessionId={statusSessionId}
          />
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
                onOpen={toggleReview}
                onOpenWorktree={openInWorktree}
                onSwitchBranch={handleSwitchBranch}
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
                  <div className="flex translate-y-[3px] items-start self-start [grid-area:menu]">{contextMenu}</div>
                  <div className="min-w-0 [grid-area:input]">{input}</div>
                  <div className="flex items-center justify-end [grid-area:controls]">{controls}</div>
                </div>
              </div>
            </div>
          </div>
        </ComposerPrimitive.Root>
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
