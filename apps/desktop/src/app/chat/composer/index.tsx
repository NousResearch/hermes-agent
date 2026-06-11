import type { Unstable_TriggerAdapter, Unstable_TriggerItem } from '@assistant-ui/core'
import { ComposerPrimitive, useAui, useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import {
  type ClipboardEvent,
  type FormEvent,
  type KeyboardEvent,
  type DragEvent as ReactDragEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'

import { hermesDirectiveFormatter, type SlashChipKind } from '@/components/assistant-ui/directive-text'
import { Button } from '@/components/ui/button'
import { useMediaQuery } from '@/hooks/use-media-query'
import { useResizeObserver } from '@/hooks/use-resize-observer'
import { useI18n } from '@/i18n'
import { chatMessageText } from '@/lib/chat-messages'
import { SLASH_COMMAND_RE } from '@/lib/chat-runtime'
import { desktopSlashCommandTakesArgs } from '@/lib/desktop-slash-commands'
import { DATA_IMAGE_URL_RE } from '@/lib/embedded-images'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'
import {
  $composerAttachments,
  clearComposerAttachments,
  clearPersistedComposerDraft,
  clearStashedComposerAttachments,
  type ComposerAttachment,
  readPersistedComposerDraft,
  stashComposerAttachments,
  takeComposerAttachments,
  writePersistedComposerDraft
} from '@/store/composer'
import {
  browseBackward,
  browseForward,
  deriveUserHistory,
  isBrowsingHistory,
  resetBrowseState
} from '@/store/composer-input-history'
import {
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  promoteQueuedPrompt,
  type QueuedPromptEntry,
  removeQueuedPrompt,
  shouldAutoDrainOnSettle,
  updateQueuedPrompt
} from '@/store/composer-queue'
import { $gatewayState, $messages, setSessionPickerOpen } from '@/store/session'
import { $threadScrolledUp } from '@/store/thread-scroll'
import { useTheme } from '@/themes'

import { extractDroppedFiles, HERMES_PATHS_MIME, partitionDroppedFiles } from '../hooks/use-composer-actions'

import { AttachmentList } from './attachments'
import { ContextMenu } from './context-menu'
import { ComposerControls } from './controls'
import { COMPOSER_DROP_ACTIVE_CLASS, COMPOSER_DROP_FADE_CLASS } from './drop-affordance'
import {
  type ComposerInsertMode,
  focusComposerInput,
  markActiveComposer,
  onComposerFocusRequest,
  onComposerInsertRefsRequest,
  onComposerInsertRequest
} from './focus'
import { HelpHint } from './help-hint'
import { useAtCompletions } from './hooks/use-at-completions'
import { useSlashCompletions } from './hooks/use-slash-completions'
import { useVoiceConversation } from './hooks/use-voice-conversation'
import { useVoiceRecorder } from './hooks/use-voice-recorder'
import {
  dragHasAttachments,
  droppedFileInlineRefs,
  type InlineRefInput,
  insertInlineRefsIntoEditor
} from './inline-refs'
import { QueuePanel } from './queue-panel'
import {
  composerPlainText,
  placeCaretEnd,
  refChipElement,
  renderComposerContents,
  RICH_INPUT_SLOT,
  slashChipElement
} from './rich-editor'
import { detectTrigger, extractClipboardImageBlobs, textBeforeCaret, type TriggerState } from './text-utils'
import { ComposerTriggerPopover } from './trigger-popover'
import type { ChatBarProps } from './types'
import { UrlDialog } from './url-dialog'
import { VoiceActivity, VoicePlaybackActivity } from './voice-activity'

const COMPOSER_STACK_BREAKPOINT_PX = 320

// A single editor line is ~28px (--composer-input-min-height 1.625rem + 0.5rem
// vertical padding). Anything taller means the text wrapped to a second line,
// which is when the composer should expand to the stacked layout.
const COMPOSER_SINGLE_LINE_MAX_PX = 36

const COMPOSER_FADE_BACKGROUND =
  'linear-gradient(to bottom, transparent, color-mix(in srgb, var(--dt-background) 10%, transparent))'

const pickPlaceholder = (pool: readonly string[]) => pool[Math.floor(Math.random() * pool.length)]

/** Completion items can carry an `action` (set in use-slash-completions) that
 *  runs a side effect on pick instead of inserting a chip — e.g. the session
 *  picker's "Browse all…" entry opens the overlay. Table-driven so new action
 *  items are a registry row, not a composer branch. */
const COMPLETION_ACTIONS: Record<string, () => void> = {
  'session-picker': () => setSessionPickerOpen(true)
}

/** Map a picked `/` completion to its pill accent. Driven by the completion
 *  group set in use-slash-completions (Skills / Themes / Commands|Options). */
function slashChipKindForItem(item: Unstable_TriggerItem): SlashChipKind {
  const group = (item.metadata as { group?: unknown } | undefined)?.group

  if (group === 'Skills') {
    return 'skill'
  }

  if (group === 'Themes') {
    return 'theme'
  }

  return 'command'
}

interface QueueEditState {
  attachments: ComposerAttachment[]
  draft: string
  entryId: string
  sessionKey: string
}

const cloneAttachments = (attachments: ComposerAttachment[]) => attachments.map(a => ({ ...a }))

// How long the composer waits after the last keystroke before persisting the
// draft to localStorage. Scope-change/unmount flushes bypass the delay.
const DRAFT_PERSIST_DEBOUNCE_MS = 400

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
  const aui = useAui()
  const draft = useAuiState(s => s.composer.text)
  const attachments = useStore($composerAttachments)
  const queuedPromptsBySession = useStore($queuedPromptsBySession)
  const scrolledUp = useStore($threadScrolledUp)
  const sessionMessages = useStore($messages)
  const activeQueueSessionKey = queueSessionKey || sessionId || null
  const draftPersistenceScope = activeQueueSessionKey || null

  const queuedPrompts = useMemo(
    () => (activeQueueSessionKey ? (queuedPromptsBySession[activeQueueSessionKey] ?? []) : []),
    [activeQueueSessionKey, queuedPromptsBySession]
  )

  const composerRef = useRef<HTMLFormElement | null>(null)
  const composerSurfaceRef = useRef<HTMLDivElement | null>(null)
  const editorRef = useRef<HTMLDivElement | null>(null)
  const draftRef = useRef(draft)
  const previousBusyRef = useRef(busy)
  // `undefined` = no skip pending. The sentinel must be distinguishable from a
  // real scope, and `null` IS a real scope (the unsaved-new-session draft):
  // resetting to null made every persist run in a new chat match the consumed
  // sentinel and bail, so new-chat drafts were never written at all.
  const skipNextDraftPersistScopeRef = useRef<string | null | undefined>(undefined)
  const pendingDraftPersistRef = useRef<{ scope: string | null; value: string } | null>(null)
  const drainingQueueRef = useRef(false)
  const urlInputRef = useRef<HTMLInputElement | null>(null)

  const [urlOpen, setUrlOpen] = useState(false)
  const [urlValue, setUrlValue] = useState('')
  const [expanded, setExpanded] = useState(false)
  const [voiceConversationActive, setVoiceConversationActive] = useState(false)
  const [tight, setTight] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [queueEdit, setQueueEdit] = useState<QueueEditState | null>(null)
  const [focusRequestId, setFocusRequestId] = useState(0)
  const dragDepthRef = useRef(0)
  const composingRef = useRef(false) // true during IME composition (CJK input)
  const lastSpokenIdRef = useRef<string | null>(null)

  const narrow = useMediaQuery('(max-width: 30rem)')

  const { availableThemes, themeName } = useTheme()
  const at = useAtCompletions({ gateway: gateway ?? null, sessionId: sessionId ?? null, cwd: cwd ?? null })
  const slash = useSlashCompletions({ activeSkin: themeName, gateway: gateway ?? null, skinThemes: availableThemes })

  const stacked = expanded || narrow || tight
  const trimmedDraft = draft.trim()
  const hasComposerPayload = trimmedDraft.length > 0 || attachments.length > 0
  const canSubmit = busy || hasComposerPayload
  const editingQueuedPrompt = queueEdit ? (queuedPrompts.find(entry => entry.id === queueEdit.entryId) ?? null) : null
  const busyAction = busy && hasComposerPayload ? 'queue' : 'stop'

  // Steer only makes sense mid-turn, text-only (the gateway can't carry images
  // into a tool result) and never for a slash command (those execute inline).
  const canSteer =
    busy && !!onSteer && attachments.length === 0 && trimmedDraft.length > 0 && !SLASH_COMMAND_RE.test(trimmedDraft)

  const showHelpHint = draft === '?'

  const { t } = useI18n()
  const gatewayState = useStore($gatewayState)
  const newSessionPlaceholders = t.composer.newSessionPlaceholders
  const followUpPlaceholders = t.composer.followUpPlaceholders

  // Resting placeholder: a starter for brand-new sessions, a continuation for
  // existing ones. Picked once and only re-rolled when we genuinely move to a
  // *different* conversation. Critically, the first id assignment of a freshly
  // started session (null → id, on the first send) is treated as the same
  // conversation so the placeholder doesn't visibly flip mid-stream.
  const [restingPlaceholder, setRestingPlaceholder] = useState(() =>
    pickPlaceholder(sessionId ? followUpPlaceholders : newSessionPlaceholders)
  )

  const prevSessionIdRef = useRef(sessionId)

  useEffect(() => {
    const prev = prevSessionIdRef.current
    prevSessionIdRef.current = sessionId

    if (prev === sessionId) {
      return
    }

    // null → id: the new session we're already in just got persisted. Keep the
    // starter we showed instead of swapping to a follow-up under the user.
    if (prev == null && sessionId) {
      return
    }

    resetBrowseState(prev)
    setRestingPlaceholder(pickPlaceholder(sessionId ? followUpPlaceholders : newSessionPlaceholders))
  }, [followUpPlaceholders, newSessionPlaceholders, sessionId])

  // When the bar is disabled it's because the gateway isn't open. Distinguish a
  // cold start ("Starting Hermes...") from a dropped connection we're trying to
  // restore (e.g. after the Mac slept) so the stuck state reads as recoverable.
  const placeholder = disabled
    ? gatewayState === 'closed' || gatewayState === 'error'
      ? t.composer.placeholderReconnecting
      : t.composer.placeholderStarting
    : restingPlaceholder

  const focusInput = useCallback(() => {
    focusComposerInput(editorRef.current)
    markActiveComposer('main')
  }, [])

  const requestMainFocus = useCallback(() => {
    setFocusRequestId(id => id + 1)
  }, [])

  const appendExternalText = useCallback(
    (text: string, mode: ComposerInsertMode) => {
      const value = text.trim()

      if (!value) {
        return
      }

      const base = mode === 'inline' ? draftRef.current.trimEnd() : draftRef.current
      const sep = mode === 'inline' ? (base ? ' ' : '') : base && !base.endsWith('\n') ? '\n\n' : ''
      const next = `${base}${sep}${value}`

      draftRef.current = next
      aui.composer().setText(next)

      const editor = editorRef.current

      if (editor) {
        renderComposerContents(editor, next)
        placeCaretEnd(editor)
      }

      setFocusRequestId(id => id + 1)
    },
    [aui]
  )

  useEffect(() => {
    if (!disabled) {
      focusInput()
    }
  }, [disabled, focusInput, focusKey, focusRequestId])

  useEffect(() => {
    if (disabled) {
      return undefined
    }

    const offFocus = onComposerFocusRequest(target => {
      if (target === 'main') {
        setFocusRequestId(id => id + 1)
      }
    })

    const offInsert = onComposerInsertRequest(({ mode, target, text }) => {
      if (target === 'main') {
        appendExternalText(text, mode)
      }
    })

    return () => {
      offFocus()
      offInsert()
    }
  }, [appendExternalText, disabled])

  // Keep draftRef in sync with the assistant-ui composer state for callers
  // that read the latest text outside the React render cycle. We don't push
  // to `$composerDraft` per keystroke any more — nobody outside the composer
  // subscribes to it (verified by grep), and the round-trip
  // `setText` ⇄ `subscribe` ⇄ `setText` was adding two useEffects to the per-
  // keystroke critical path. `reconcileComposerTerminalSelections` only
  // matters when the draft is submitted; we now call it from the submit
  // path instead.
  useEffect(() => {
    draftRef.current = draft

    const editor = editorRef.current

    if (editor && document.activeElement !== editor && composerPlainText(editor) !== draft) {
      renderComposerContents(editor, draft)
    }
  }, [draft])

  useEffect(() => {
    if (urlOpen) {
      window.requestAnimationFrame(() => urlInputRef.current?.focus({ preventScroll: true }))
    }
  }, [urlOpen])

  // Expansion (input on its own full-width row, controls below) is driven by
  // the editor's *actual* rendered height via the ResizeObserver in
  // syncComposerMetrics — it only fires when the text genuinely wraps to a
  // second line, so the layout flips exactly at the wrap point rather than at
  // a guessed character count. We only handle the two cases the observer
  // can't: an explicit newline (expand before layout settles) and an emptied
  // draft (collapse back). We never read scrollHeight per keystroke.
  useEffect(() => {
    if (!draft) {
      setExpanded(false)

      return
    }

    if (expanded) {
      return
    }

    if (draft.includes('\n')) {
      setExpanded(true)
    }
  }, [draft, expanded])

  // Bucket measured heights so we only invalidate the global CSS var when
  // the size crosses a meaningful threshold. Without bucketing, the editor
  // grows ~1px per character → setProperty fires every keystroke → entire
  // tree's computed style is invalidated → next paint forces a full
  // recalculate-style pass. With an 8px bucket, the invalidation rate drops
  // ~8× and small char-by-char typing produces no style invalidation at all
  // until a wrap or row change actually happens.
  const lastBucketedHeightRef = useRef(0)
  const lastBucketedSurfaceHeightRef = useRef(0)
  const lastTightRef = useRef<boolean | null>(null)

  const syncComposerMetrics = useCallback(() => {
    const composer = composerRef.current

    if (!composer) {
      return
    }

    const { height, width } = composer.getBoundingClientRect()
    const surfaceHeight = composerSurfaceRef.current?.getBoundingClientRect().height
    const root = document.documentElement

    if (width > 0) {
      const nextTight = width < COMPOSER_STACK_BREAKPOINT_PX

      if (nextTight !== lastTightRef.current) {
        lastTightRef.current = nextTight
        setTight(nextTight)
      }
    }

    // Expand once the input has actually wrapped past a single line. The
    // observer only fires on real size changes, so this reads scrollHeight at
    // most once per wrap (not per keystroke). One line ≈ 28px (1.625rem
    // min-height + padding); a second line clears ~36px. We only ever expand
    // here — collapse is handled by the emptied-draft effect to avoid
    // oscillating across the wrap boundary as the input switches widths.
    const editor = editorRef.current

    if (editor && editor.scrollHeight > COMPOSER_SINGLE_LINE_MAX_PX) {
      setExpanded(true)
    }

    if (height > 0) {
      const bucket = Math.round(height / 8) * 8

      if (bucket !== lastBucketedHeightRef.current) {
        lastBucketedHeightRef.current = bucket
        root.style.setProperty('--composer-measured-height', `${bucket}px`)
      }
    }

    if (surfaceHeight && surfaceHeight > 0) {
      const bucket = Math.round(surfaceHeight / 8) * 8

      if (bucket !== lastBucketedSurfaceHeightRef.current) {
        lastBucketedSurfaceHeightRef.current = bucket
        root.style.setProperty('--composer-surface-measured-height', `${bucket}px`)
      }
    }
  }, [])

  useResizeObserver(syncComposerMetrics, composerRef, composerSurfaceRef, editorRef)

  useEffect(() => {
    return () => {
      const root = document.documentElement
      root.style.removeProperty('--composer-measured-height')
      root.style.removeProperty('--composer-surface-measured-height')
    }
  }, [])

  const insertText = (text: string) => {
    const currentDraft = draftRef.current
    const sep = currentDraft && !currentDraft.endsWith('\n') ? '\n' : ''
    const nextDraft = `${currentDraft}${sep}${text}`

    draftRef.current = nextDraft
    aui.composer().setText(nextDraft)

    // Push the new text into the contentEditable editor directly. Setting the
    // assistant-ui composer state alone is not enough: the draft→editor sync
    // effect only re-renders the editor when it is NOT focused
    // (document.activeElement !== editor), and the dictation/insert paths
    // typically run while the editor has (or immediately regains) focus — so
    // the store would hold the text but the visible editor would stay empty
    // and there'd be nothing to send. Mirror appendExternalText here.
    const editor = editorRef.current

    if (editor) {
      renderComposerContents(editor, nextDraft)
      placeCaretEnd(editor)
    }

    requestMainFocus()
  }

  const insertInlineRefs = (refs: InlineRefInput[]) => {
    const editor = editorRef.current

    if (!editor) {
      return false
    }

    const nextDraft = insertInlineRefsIntoEditor(editor, refs)

    if (nextDraft === null) {
      return false
    }

    draftRef.current = nextDraft
    aui.composer().setText(nextDraft)
    requestMainFocus()

    return true
  }

  // Latest-closure ref so the (once-only) subscription always calls the current
  // insertInlineRefs without re-subscribing every render.
  const insertInlineRefsRef = useRef(insertInlineRefs)
  insertInlineRefsRef.current = insertInlineRefs

  useEffect(() => {
    return onComposerInsertRefsRequest(({ refs, target }) => {
      if (target === 'main') {
        insertInlineRefsRef.current(refs)
      }
    })
  }, [])

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

      return
    }

    if (DATA_IMAGE_URL_RE.test(pastedText)) {
      event.preventDefault()

      return
    }

    event.preventDefault()
    document.execCommand('insertText', false, pastedText)
    const nextDraft = composerPlainText(event.currentTarget)
    draftRef.current = nextDraft
    aui.composer().setText(nextDraft)
  }

  const [trigger, setTrigger] = useState<TriggerState | null>(null)
  const [triggerActive, setTriggerActive] = useState(0)
  const [triggerItems, setTriggerItems] = useState<readonly Unstable_TriggerItem[]>([])
  // Set synchronously in keydown when the open trigger popover consumes a
  // navigation/control key (Arrow/Enter/Tab/Escape). The subsequent keyup must
  // NOT run refreshTrigger for that keypress: it never edits text, and for
  // Escape the keydown has already set trigger=null, so a keyup refresh would
  // re-detect the still-present `/` and instantly reopen the menu. A ref is
  // used instead of reading `trigger` in keyup because by keyup time React has
  // re-rendered and the handler closure sees the post-keydown state.
  const triggerKeyConsumedRef = useRef(false)

  const refreshTrigger = useCallback(() => {
    const editor = editorRef.current

    if (!editor) {
      return
    }

    // Fast-bail: if neither `@` nor `/` appears in the current draft, there's
    // nothing for `detectTrigger` to match. Use `textContent` (cheap browser-
    // native walk) for the precondition check rather than `composerPlainText`
    // (recursive child walk with chip-aware logic). Only when a trigger char
    // is present do we pay the cost of the full walk + DOM range work.
    const rawText = editor.textContent ?? ''

    if (!rawText.includes('@') && !rawText.includes('/')) {
      if (trigger) {
        setTrigger(null)
        setTriggerActive(0)
      }

      return
    }

    const before = textBeforeCaret(editor)
    const detected = detectTrigger(before ?? composerPlainText(editor))

    setTrigger(detected)

    // Only reset the highlight when the trigger actually changed (opened, or
    // the query/kind differs). Re-detecting the *same* trigger — e.g. on a
    // caret move (mouseup) or a stray refresh — must preserve the user's
    // current selection instead of snapping back to the first item.
    if (detected?.kind !== trigger?.kind || detected?.query !== trigger?.query) {
      setTriggerActive(0)
    }
  }, [trigger])

  // Pull the live contentEditable text into draftRef + the AUI composer state
  // (which drives `hasComposerPayload` → the send button). Shared by the input
  // and compositionend paths so committed IME text reaches state through either.
  const flushEditorToDraft = (editor: HTMLDivElement) => {
    if (editor.childNodes.length === 1 && editor.firstChild?.nodeName === 'BR') {
      editor.replaceChildren()
    }

    const nextDraft = composerPlainText(editor)

    if (nextDraft !== draftRef.current) {
      draftRef.current = nextDraft
      aui.composer().setText(nextDraft)
    }

    window.setTimeout(refreshTrigger, 0)
  }

  const handleEditorInput = (event: FormEvent<HTMLDivElement>) => {
    // During IME composition the DOM contains uncommitted preedit text
    // mixed with real content.  Skip state writes — compositionend flushes
    // the finalized text (see onCompositionEnd).
    if (composingRef.current) {
      return
    }

    flushEditorToDraft(event.currentTarget)
  }

  const triggerAdapter: Unstable_TriggerAdapter | null =
    trigger?.kind === '@' ? at.adapter : trigger?.kind === '/' ? slash.adapter : null

  useEffect(() => {
    if (!trigger || !triggerAdapter?.search) {
      setTriggerItems([])

      return
    }

    setTriggerItems(triggerAdapter.search(trigger.query))
  }, [trigger, triggerAdapter])

  const triggerLoading = trigger?.kind === '@' ? at.loading : trigger?.kind === '/' ? slash.loading : false

  const closeTrigger = () => {
    setTrigger(null)
    setTriggerItems([])
    setTriggerActive(0)
  }

  useEffect(() => {
    setTriggerActive(idx => Math.min(idx, Math.max(0, triggerItems.length - 1)))
  }, [triggerItems.length])

  const replaceTriggerWithChip = (item: Unstable_TriggerItem) => {
    const editor = editorRef.current

    if (!editor || !trigger) {
      return
    }

    // Action items (e.g. "Browse all sessions…") run a side effect instead of
    // inserting a chip: strip the typed trigger token, then fire the action.
    const completionAction = (item.metadata as { action?: unknown } | undefined)?.action
    const runAction = typeof completionAction === 'string' ? COMPLETION_ACTIONS[completionAction] : undefined

    if (runAction) {
      const current = composerPlainText(editor)
      const prefix = current.slice(0, Math.max(0, current.length - trigger.tokenLength))

      renderComposerContents(editor, prefix)
      placeCaretEnd(editor)
      draftRef.current = composerPlainText(editor)
      aui.composer().setText(draftRef.current)
      closeTrigger()
      runAction()
      requestMainFocus()

      return
    }

    const serialized = hermesDirectiveFormatter.serialize(item)
    const starter = serialized.endsWith(':')

    // Picking a bare arg-taking command (e.g. `/personality`) shouldn't commit
    // it — expand to its options step so the popover shows the inline list, just
    // as typing `/personality ` by hand would. A serialized value with a space is
    // already an arg pick (`/personality alice`), so it commits normally.
    const command = (item.metadata as { command?: string } | undefined)?.command ?? ''

    const expandsToArgs =
      trigger.kind === '/' && !serialized.includes(' ') && desktopSlashCommandTakesArgs(command)

    const text = starter || serialized.endsWith(' ') ? serialized : `${serialized} `
    const directive = !starter && serialized.match(/^@([^:]+):(.+)$/)
    // No pill while expanding — the bare command stays plain text until an arg
    // is picked, at which point a single pill is emitted for the full command.
    const slashKind = !expandsToArgs && trigger.kind === '/' ? slashChipKindForItem(item) : null
    const keepTriggerOpen = starter || expandsToArgs

    const finish = () => {
      draftRef.current = composerPlainText(editor)
      aui.composer().setText(draftRef.current)
      requestMainFocus()
      keepTriggerOpen ? window.setTimeout(refreshTrigger, 0) : closeTrigger()
    }

    const sel = window.getSelection()
    const range = sel?.rangeCount ? sel.getRangeAt(0) : null
    const node = range?.startContainer
    const offset = range?.startOffset ?? 0

    if (!sel || !range || node?.nodeType !== Node.TEXT_NODE || offset < trigger.tokenLength) {
      const current = composerPlainText(editor)
      const prefix = current.slice(0, Math.max(0, current.length - trigger.tokenLength))

      if (slashKind) {
        // Two-step arg picks (e.g. `/handoff` pill already inserted, now picking
        // the platform) land here because the caret sits past a contenteditable
        // chip. Rebuild the prefix and re-emit a single pill for the full command.
        renderComposerContents(editor, prefix)
        editor.append(slashChipElement(serialized, slashKind), document.createTextNode(' '))
        placeCaretEnd(editor)

        return finish()
      }

      renderComposerContents(editor, `${prefix}${text}`)
      placeCaretEnd(editor)

      return finish()
    }

    const replaceRange = document.createRange()
    replaceRange.setStart(node, offset - trigger.tokenLength)
    replaceRange.setEnd(node, offset)
    replaceRange.deleteContents()

    const chip = slashKind
      ? slashChipElement(serialized, slashKind)
      : directive
        ? refChipElement(directive[1], directive[2])
        : null

    if (chip) {
      const space = document.createTextNode(' ')
      const fragment = document.createDocumentFragment()
      fragment.append(chip, space)
      replaceRange.insertNode(fragment)

      const caret = document.createRange()
      caret.setStart(space, 1)
      caret.collapse(true)
      sel.removeAllRanges()
      sel.addRange(caret)

      return finish()
    }

    document.execCommand('insertText', false, text)
    finish()
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

      if (event.key === 'Enter' || event.key === 'Tab') {
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

      const history = deriveUserHistory(sessionMessages, chatMessageText)
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

        const history = deriveUserHistory(sessionMessages, chatMessageText)
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

      // Otherwise Esc interrupts the running turn (Stop-button parity).
      if (busy) {
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

  const resetDragState = () => {
    dragDepthRef.current = 0
    setDragActive(false)
  }

  const handleDragEnter = (event: ReactDragEvent<HTMLFormElement>) => {
    if (!onAttachDroppedItems || !dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME)) {
      return
    }

    event.preventDefault()
    dragDepthRef.current += 1

    if (!dragActive) {
      setDragActive(true)
    }
  }

  const handleDragOver = (event: ReactDragEvent<HTMLFormElement>) => {
    if (!onAttachDroppedItems || !dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME)) {
      return
    }

    event.preventDefault()
    event.dataTransfer.dropEffect = 'copy'
  }

  const handleDragLeave = (event: ReactDragEvent<HTMLFormElement>) => {
    if (!onAttachDroppedItems) {
      return
    }

    event.preventDefault()
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1)

    if (dragDepthRef.current === 0) {
      setDragActive(false)
    }
  }

  const handleDrop = (event: ReactDragEvent<HTMLFormElement>) => {
    if (!onAttachDroppedItems) {
      return
    }

    event.preventDefault()
    resetDragState()

    const candidates = extractDroppedFiles(event.dataTransfer)

    if (candidates.length === 0) {
      return
    }

    // In-app drags (project tree / gutter) are workspace-relative paths the
    // gateway resolves directly, so they stay inline @file:/@line: refs. OS
    // drops are absolute local paths a remote gateway can't read (and images
    // need byte upload for vision), so route them through the upload pipeline.
    const { inAppRefs, osDrops } = partitionDroppedFiles(candidates)
    const refs = droppedFileInlineRefs(inAppRefs, cwd)

    if (refs.length && insertInlineRefs(refs)) {
      triggerHaptic('selection')
    }

    if (osDrops.length) {
      void Promise.resolve(onAttachDroppedItems(osDrops)).then(attached => {
        if (attached) {
          triggerHaptic('selection')
          requestMainFocus()
        }
      })
    }
  }

  const handleInputDragOver = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME)) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    event.dataTransfer.dropEffect = 'copy'
  }

  const handleInputDrop = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME)) {
      return
    }

    const candidates = extractDroppedFiles(event.dataTransfer)

    if (!candidates.length) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    resetDragState()

    // Dropping straight onto the text box used to inline-ref *every* file —
    // including OS/Finder drops, whose absolute local path a remote gateway
    // can't read and whose image bytes never reached vision. Split by origin:
    // in-app drags stay inline refs; OS drops go through the upload pipeline.
    // (When no upload handler is wired, fall back to inline refs for all.)
    const attach = onAttachDroppedItems
    const { inAppRefs, osDrops } = partitionDroppedFiles(candidates)
    const refs = droppedFileInlineRefs(attach ? inAppRefs : candidates, cwd)

    if (refs.length && insertInlineRefs(refs)) {
      triggerHaptic('selection')
    }

    if (attach && osDrops.length) {
      void Promise.resolve(attach(osDrops)).then(attached => {
        if (attached) {
          triggerHaptic('selection')
          requestMainFocus()
        }
      })
    }
  }

  const clearDraft = useCallback(() => {
    aui.composer().setText('')
    draftRef.current = ''

    if (editorRef.current) {
      editorRef.current.replaceChildren()
    }
  }, [aui])

  const loadIntoComposer = (text: string, attachments: ComposerAttachment[]) => {
    draftRef.current = text
    aui.composer().setText(text)
    $composerAttachments.set(cloneAttachments(attachments))

    const editor = editorRef.current

    if (editor) {
      renderComposerContents(editor, text)
      placeCaretEnd(editor)
    }
  }

  // Restore a scope's draft (persisted text + in-memory attachments) when we
  // enter it, and stash the attachments back when we leave. Text rides through
  // localStorage so it survives reloads; attachments carry live blobs/upload
  // state that can't serialize, so they're retained in memory only — enough to
  // survive a session switch, which is the case users actually hit.
  useEffect(() => {
    const persisted = readPersistedComposerDraft(draftPersistenceScope)
    const restoredAttachments = takeComposerAttachments(draftPersistenceScope)
    skipNextDraftPersistScopeRef.current = draftPersistenceScope
    loadIntoComposer(persisted, restoredAttachments)

    return () => {
      stashComposerAttachments(draftPersistenceScope, $composerAttachments.get())
    }
  }, [draftPersistenceScope]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (skipNextDraftPersistScopeRef.current === draftPersistenceScope) {
      skipNextDraftPersistScopeRef.current = undefined

      return
    }

    // Don't persist programmatically-loaded text: browsing sent-message
    // history or editing a queued prompt swaps the composer to recalled text,
    // and persisting that would clobber the genuine in-progress draft (which
    // history keeps in its own snapshot and restores on the way back). Leaving
    // the prior pending write untouched keeps the real draft in storage.
    if (isBrowsingHistory(sessionId) || queueEdit) {
      return
    }

    // Debounce the localStorage write: the composer's per-keystroke path was
    // deliberately slimmed down (see the draftRef sync comment above), so we
    // don't touch storage on every keypress. The pending ref below is flushed
    // on scope change / unmount so a fast session switch can't drop the
    // trailing keystrokes.
    pendingDraftPersistRef.current = { scope: draftPersistenceScope, value: draft }

    const handle = window.setTimeout(() => {
      pendingDraftPersistRef.current = null
      writePersistedComposerDraft(draftPersistenceScope, draft)
    }, DRAFT_PERSIST_DEBOUNCE_MS)

    return () => window.clearTimeout(handle)
  }, [draft, draftPersistenceScope, queueEdit, sessionId])

  // Flush any pending debounced draft write when leaving a session scope,
  // unmounting, or the window unloading, so the latest text is always
  // persisted. The pagehide listener is load-bearing: React does NOT run
  // effect cleanups on a page reload, so without it a Cmd+R inside the
  // debounce window silently dropped everything typed in the last 400ms.
  useEffect(() => {
    const flushPendingDraftPersist = () => {
      const pending = pendingDraftPersistRef.current

      if (pending) {
        pendingDraftPersistRef.current = null
        writePersistedComposerDraft(pending.scope, pending.value)
      }
    }

    window.addEventListener('pagehide', flushPendingDraftPersist)

    return () => {
      window.removeEventListener('pagehide', flushPendingDraftPersist)
      flushPendingDraftPersist()
    }
  }, [draftPersistenceScope])

  const beginQueuedEdit = (entry: QueuedPromptEntry) => {
    if (!activeQueueSessionKey || queueEdit) {
      return
    }

    setQueueEdit({
      attachments: cloneAttachments($composerAttachments.get()),
      draft: draftRef.current,
      entryId: entry.id,
      sessionKey: activeQueueSessionKey
    })
    loadIntoComposer(entry.text, entry.attachments)
    triggerHaptic('selection')
    focusInput()
  }

  // Walk queued entries while editing (ArrowUp = older, ArrowDown = newer),
  // saving the in-progress edit on each step. Stepping newer past the last
  // entry exits edit mode and restores the pre-edit draft.
  const stepQueuedEdit = (direction: -1 | 1) => {
    if (!queueEdit) {
      return false
    }

    const index = queuedPrompts.findIndex(e => e.id === queueEdit.entryId)
    const target = index + direction

    if (index < 0 || target < 0) {
      return index >= 0 // at the oldest: swallow; missing entry: let it fall through
    }

    const saved = updateQueuedPrompt(queueEdit.sessionKey, queueEdit.entryId, {
      attachments: cloneAttachments($composerAttachments.get()),
      text: draftRef.current
    })

    const next = queuedPrompts[target]

    if (next) {
      setQueueEdit({ ...queueEdit, entryId: next.id })
      loadIntoComposer(next.text, next.attachments)
    } else {
      setQueueEdit(null)
      loadIntoComposer(queueEdit.draft, queueEdit.attachments)
    }

    triggerHaptic(saved ? 'success' : 'selection')
    focusInput()

    return true
  }

  const exitQueuedEdit = (action: 'cancel' | 'save'): boolean => {
    if (!queueEdit) {
      return false
    }

    if (action === 'save') {
      const text = draftRef.current
      const next = cloneAttachments($composerAttachments.get())

      if (!text.trim() && next.length === 0) {
        return false
      }

      const saved = updateQueuedPrompt(queueEdit.sessionKey, queueEdit.entryId, { attachments: next, text })
      triggerHaptic(saved ? 'success' : 'selection')
    } else {
      triggerHaptic('cancel')
    }

    loadIntoComposer(queueEdit.draft, queueEdit.attachments)
    setQueueEdit(null)
    focusInput()

    return true
  }

  const queueCurrentDraft = useCallback(() => {
    if (!activeQueueSessionKey || (!draft.trim() && attachments.length === 0)) {
      return false
    }

    if (!enqueueQueuedPrompt(activeQueueSessionKey, { text: draft, attachments })) {
      return false
    }

    clearDraft()
    clearComposerAttachments()
    triggerHaptic('selection')

    return true
  }, [activeQueueSessionKey, attachments, clearDraft, draft])

  // Steer the live turn (nudge without interrupting). Clears the draft up front
  // for snappy feedback; if the gateway rejects (no live tool window) the words
  // are re-queued so nothing is lost — same safety net as a plain queue.
  const steerDraft = useCallback(() => {
    if (!onSteer || !canSteer) {
      return
    }

    const text = draftRef.current.trim()

    triggerHaptic('submit')
    clearDraft()

    void Promise.resolve(onSteer(text)).then(accepted => {
      if (!accepted && activeQueueSessionKey) {
        enqueueQueuedPrompt(activeQueueSessionKey, { text, attachments: [] })
      }
    })
  }, [activeQueueSessionKey, canSteer, clearDraft, onSteer])

  // All queue drain paths share one lock + send-then-remove sequence.
  // `pickEntry` lets each caller choose head, by-id, or skip-edited.
  const runDrain = useCallback(
    async (pickEntry: (entries: QueuedPromptEntry[]) => QueuedPromptEntry | undefined): Promise<boolean> => {
      if (drainingQueueRef.current || !activeQueueSessionKey) {
        return false
      }

      const entry = pickEntry(queuedPrompts)

      if (!entry) {
        return false
      }

      drainingQueueRef.current = true

      try {
        const accepted = await Promise.resolve(
          onSubmit(entry.text, { attachments: entry.attachments, fromQueue: true })
        )

        if (accepted === false) {
          return false
        }

        removeQueuedPrompt(activeQueueSessionKey, entry.id)
        resetBrowseState(sessionId)

        return true
      } finally {
        drainingQueueRef.current = false
      }
    },
    [activeQueueSessionKey, onSubmit, queuedPrompts, sessionId]
  )

  const drainNextQueued = useCallback(
    () =>
      runDrain(entries => {
        const skip = queueEdit?.entryId

        return skip ? entries.find(e => e.id !== skip) : entries[0]
      }),
    [queueEdit, runDrain]
  )

  const sendQueuedNow = useCallback(
    (id: string) => {
      if (!activeQueueSessionKey || id === queueEdit?.entryId) {
        return false
      }

      if (busy) {
        // Promote to the head, then interrupt. The gateway always emits a
        // settle (message.complete + session.info running:false) when the
        // turn unwinds, and the busy→false auto-drain below sends this entry.
        promoteQueuedPrompt(activeQueueSessionKey, id)
        triggerHaptic('selection')
        void Promise.resolve(onCancel())

        return true
      }

      return runDrain(entries => entries.find(e => e.id === id))
    },
    [activeQueueSessionKey, busy, onCancel, queueEdit, runDrain]
  )

  // Auto-drain on busy → false (turn settled). Queued turns always flow once
  // the session is idle again — whether the turn finished naturally or the
  // user interrupted it. Interrupting to reach a queued message is the whole
  // point of the queue, so we never suppress the drain. To cancel queued
  // turns, the user deletes them from the panel.
  useEffect(() => {
    const wasBusy = previousBusyRef.current
    previousBusyRef.current = busy

    if (
      shouldAutoDrainOnSettle({
        isBusy: busy,
        queueLength: queuedPrompts.length,
        wasBusy
      })
    ) {
      void drainNextQueued()
    }
  }, [busy, drainNextQueued, queuedPrompts.length])

  // Clean up queue edit when its target disappears (session swap or external delete).
  useEffect(() => {
    if (!queueEdit) {
      return
    }

    if (queueEdit.sessionKey === activeQueueSessionKey && editingQueuedPrompt) {
      return
    }

    loadIntoComposer(queueEdit.draft, queueEdit.attachments)
    setQueueEdit(null)
  }, [activeQueueSessionKey, editingQueuedPrompt, queueEdit]) // eslint-disable-line react-hooks/exhaustive-deps

  const submitDraft = () => {
    // Source the text from the DOM editor, not React state. The AUI composer
    // state (`draft`) and the derived `hasComposerPayload` lag the DOM by a
    // render, so on fast typing or IME composition the final keystroke(s) may
    // not have synced yet — reading state here drops the message (Enter looks
    // like it does nothing; typing a trailing space only "fixes" it because the
    // extra input event forces a state sync). draftRef is updated on every
    // input event; refresh it from the editor once more to also cover an
    // in-flight keystroke that hasn't fired its input event yet.
    const editor = editorRef.current

    if (editor) {
      const domText = composerPlainText(editor)

      if (domText !== draftRef.current) {
        draftRef.current = domText
        aui.composer().setText(domText)
      }
    }

    const text = draftRef.current
    const payloadPresent = text.trim().length > 0 || attachments.length > 0

    if (queueEdit) {
      exitQueuedEdit('save')
    } else if (busy) {
      // Slash commands should execute immediately even while the agent is
      // busy — they're client-side operations (/yolo, /skin, /new, /help,
      // etc.) or self-contained gateway RPCs (/status, /compress).  onSubmit
      // routes them to executeSlashCommand, which has its own per-command
      // busy guard for commands that genuinely need an idle session (skill
      // /send directives).  Queuing them would make every slash command wait
      // for the current turn to finish, which is how the TUI never behaves.
      if (!attachments.length && SLASH_COMMAND_RE.test(text.trim())) {
        const submitted = text
        triggerHaptic('submit')
        clearDraft()
        void Promise.resolve(onSubmit(submitted)).then(accepted => {
          if (accepted === false) {
            loadIntoComposer(submitted, [])
            writePersistedComposerDraft(draftPersistenceScope, submitted)
          } else {
            clearPersistedComposerDraft(draftPersistenceScope)
            clearStashedComposerAttachments(draftPersistenceScope)
          }
        }).catch(() => {
          loadIntoComposer(submitted, [])
          writePersistedComposerDraft(draftPersistenceScope, submitted)
        })
      } else if (payloadPresent) {
        queueCurrentDraft()
      } else {
        // Stop button (the only way to reach here while busy with an empty
        // composer — empty Enter is short-circuited in the keydown handler).
        triggerHaptic('cancel')
        void Promise.resolve(onCancel())
      }
    } else if (!payloadPresent && queuedPrompts.length > 0) {
      void drainNextQueued()
    } else if (payloadPresent) {
      const submitted = text
      const submittedAttachments = cloneAttachments(attachments)
      triggerHaptic('submit')
      resetBrowseState(sessionId)
      clearDraft()
      clearComposerAttachments()
      void Promise.resolve(onSubmit(submitted, { attachments: submittedAttachments })).then(accepted => {
        if (accepted === false) {
          loadIntoComposer(submitted, submittedAttachments)
          writePersistedComposerDraft(draftPersistenceScope, submitted)
        } else {
          clearPersistedComposerDraft(draftPersistenceScope)
          clearStashedComposerAttachments(draftPersistenceScope)
        }
      }).catch(() => {
        loadIntoComposer(submitted, submittedAttachments)
        writePersistedComposerDraft(draftPersistenceScope, submitted)
      })
    }

    focusInput()
  }

  const submitUrl = () => {
    const url = urlValue.trim()

    if (!url) {
      return
    }

    if (onAddUrl) {
      onAddUrl(url)
    } else {
      insertText(`@url:${url}`)
    }

    triggerHaptic('success')
    setUrlValue('')
    setUrlOpen(false)
  }

  const { dictate, voiceActivityState, voiceStatus } = useVoiceRecorder({
    focusInput,
    maxRecordingSeconds,
    onTranscript: insertText,
    onTranscribeAudio
  })

  const pendingResponse = () => {
    const messages = $messages.get()
    const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)

    if (!last || last.id === lastSpokenIdRef.current) {
      return null
    }

    const text = chatMessageText(last).trim()

    if (!text) {
      return null
    }

    return {
      id: last.id,
      pending: Boolean(last.pending),
      text
    }
  }

  const consumePendingResponse = () => {
    const messages = $messages.get()
    const last = messages.findLast(m => m.role === 'assistant' && !m.hidden)

    if (last) {
      lastSpokenIdRef.current = last.id
    }
  }

  const submitVoiceTurn = async (text: string) => {
    if (busy) {
      return
    }

    triggerHaptic('submit')
    resetBrowseState(sessionId)
    clearDraft()
    await onSubmit(text)
  }

  const conversation = useVoiceConversation({
    busy,
    consumePendingResponse,
    enabled: voiceConversationActive,
    onFatalError: () => setVoiceConversationActive(false),
    onSubmit: submitVoiceTurn,
    onTranscribeAudio,
    pendingResponse
  })

  const contextMenu = (
    <ContextMenu
      onInsertText={insertText}
      onOpenUrlDialog={() => {
        triggerHaptic('open')
        setUrlOpen(true)
      }}
      onPasteClipboardImage={onPasteClipboardImage}
      onPickFiles={onPickFiles}
      onPickFolders={onPickFolders}
      onPickImages={onPickImages}
      state={state}
    />
  )

  const controls = (
    <ComposerControls
      busy={busy}
      busyAction={busyAction}
      canSteer={canSteer}
      canSubmit={canSubmit}
      conversation={{
        active: voiceConversationActive,
        level: conversation.level,
        muted: conversation.muted,
        onEnd: () => {
          setVoiceConversationActive(false)
          void conversation.end()
        },
        onStart: () => setVoiceConversationActive(true),
        onStopTurn: conversation.stopTurn,
        onToggleMute: conversation.toggleMute,
        status: conversation.status
      }}
      disabled={disabled}
      hasComposerPayload={hasComposerPayload}
      onDictate={dictate}
      onSteer={steerDraft}
      state={state}
      voiceStatus={voiceStatus}
    />
  )

  const input = (
    <div className={cn('relative', stacked ? 'w-full' : 'min-w-(--composer-input-inline-min-width) flex-1')}>
      <div
        aria-label={t.composer.message}
        autoCapitalize="off"
        autoCorrect="off"
        className={cn(
          'min-h-(--composer-input-min-height) max-h-(--composer-input-max-height) overflow-y-auto whitespace-pre-wrap break-words [overflow-wrap:anywhere] bg-transparent pb-1 pr-1 pt-1 leading-normal text-foreground outline-none disabled:cursor-not-allowed',
          'empty:before:content-[attr(data-placeholder)] empty:before:text-muted-foreground/60',
          '**:data-ref-text:cursor-default',
          stacked && 'pl-3',
          stacked ? 'w-full' : 'min-w-(--composer-input-inline-min-width) flex-1'
        )}
        contentEditable={!disabled}
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
        spellCheck="true"
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
        <textarea aria-hidden className="sr-only" tabIndex={-1} />
      </ComposerPrimitive.Input>
    </div>
  )

  return (
    <>
      <ComposerPrimitive.Unstable_TriggerPopoverRoot>
        <ComposerPrimitive.Root
          className="group/composer absolute bottom-0 left-1/2 z-30 w-[min(var(--composer-width),calc(100%-2rem))] max-w-full -translate-x-1/2 rounded-2xl pt-2 pb-[var(--composer-shell-pad-block-end)]"
          data-drag-active={dragActive ? '' : undefined}
          data-slot="composer-root"
          data-thread-scrolled-up={scrolledUp ? '' : undefined}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onSubmit={e => {
            e.preventDefault()

            if (composingRef.current) {
              return
            }

            submitDraft()
          }}
          ref={composerRef}
        >
          {showHelpHint && <HelpHint />}
          {trigger && (
            <ComposerTriggerPopover
              activeIndex={triggerActive}
              items={triggerItems}
              kind={trigger.kind}
              loading={triggerLoading}
              onHover={setTriggerActive}
              onPick={replaceTriggerWithChip}
            />
          )}
          {activeQueueSessionKey && queuedPrompts.length > 0 && (
            // Out of flow so the queue never inflates the composer's measured
            // height (that drives thread bottom padding → chat resizes on
            // queue). Overlaps -mb-2 onto the surface's top border for a shared
            // edge; capped + scrollable. Overlays the chat instead of pushing it.
            <div className="absolute inset-x-0 bottom-full z-6 -mb-2 max-h-[40vh] overflow-y-auto">
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
            </div>
          )}
          <div
            className="pointer-events-none absolute inset-0 rounded-[inherit]"
            style={{ background: COMPOSER_FADE_BACKGROUND }}
          />
          <div className="relative w-full rounded-[inherit]">
            <div
              className={cn(
                'relative z-4 isolate rounded-[inherit] border border-[color-mix(in_srgb,var(--dt-composer-ring)_calc(18%*var(--composer-ring-strength)),var(--dt-input))] transition-[border-color] duration-200 ease-out',
                COMPOSER_DROP_FADE_CLASS,
                'group-focus-within/composer:border-[color-mix(in_srgb,var(--dt-composer-ring)_calc(45%*var(--composer-ring-strength)),transparent)]',
                'group-has-data-[state=open]/composer:border-t-transparent',
                dragActive && COMPOSER_DROP_ACTIVE_CLASS
              )}
              data-slot="composer-surface"
              ref={composerSurfaceRef}
            >
              <div
                aria-hidden
                className={cn(
                  'pointer-events-none absolute inset-0 -z-10 rounded-[inherit]',
                  'bg-[color-mix(in_srgb,var(--dt-card)_72%,transparent)]',
                  'backdrop-blur-[0.75rem] backdrop-saturate-[1.12]',
                  '[-webkit-backdrop-filter:blur(0.75rem)_saturate(1.12)]',
                  'transition-[background-color] duration-150 ease-out',
                  'group-data-[thread-scrolled-up]/composer:bg-[color-mix(in_srgb,var(--dt-card)_48%,transparent)]',
                  'group-focus-within/composer:bg-[color-mix(in_srgb,var(--dt-card)_85%,transparent)]'
                )}
              />
              <div
                className={cn(
                  'relative z-1 flex min-h-0 w-full flex-col gap-(--composer-row-gap) overflow-hidden rounded-[inherit] px-(--composer-surface-pad-x) py-(--composer-surface-pad-y) transition-opacity duration-200 ease-out',
                  scrolledUp
                    ? 'opacity-30 group-hover/composer:opacity-100 group-focus-within/composer:opacity-100'
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
                  <div className="flex items-center [grid-area:menu]">{contextMenu}</div>
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
            'bg-[color-mix(in_srgb,var(--dt-card)_72%,transparent)]',
            'backdrop-blur-[0.75rem] backdrop-saturate-[1.12]',
            '[-webkit-backdrop-filter:blur(0.75rem)_saturate(1.12)]',
            'transition-[background-color] duration-150 ease-out',
            'group-data-[thread-scrolled-up]/composer:bg-[color-mix(in_srgb,var(--dt-card)_48%,transparent)]',
            'group-focus-within/composer:bg-[color-mix(in_srgb,var(--dt-card)_85%,transparent)]'
          )}
        />
      </div>
    </div>
  )
}
