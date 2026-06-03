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

import { hermesDirectiveFormatter } from '@/components/assistant-ui/directive-text'
import { Button } from '@/components/ui/button'
import { useMediaQuery } from '@/hooks/use-media-query'
import { useResizeObserver } from '@/hooks/use-resize-observer'
import { chatMessageText } from '@/lib/chat-messages'
import { DATA_IMAGE_URL_RE } from '@/lib/embedded-images'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'
import {
  $composerAttachments,
  clearComposerAttachments,
  type ComposerAttachment
} from '@/store/composer'
import {
  $queuedPromptsBySession,
  enqueueQueuedPrompt,
  type QueuedPromptEntry,
  removeQueuedPrompt,
  updateQueuedPrompt
} from '@/store/composer-queue'
import { $messages } from '@/store/session'
import { $threadScrolledUp } from '@/store/thread-scroll'

import { extractDroppedFiles, HERMES_PATHS_MIME } from '../hooks/use-composer-actions'

import { AttachmentList } from './attachments'
import { ContextMenu } from './context-menu'
import { ComposerControls } from './controls'
import { COMPOSER_DROP_ACTIVE_CLASS, COMPOSER_DROP_FADE_CLASS } from './drop-affordance'
import {
  type ComposerInsertMode,
  focusComposerInput,
  markActiveComposer,
  onComposerFocusRequest,
  onComposerInsertRequest
} from './focus'
import { HelpHint } from './help-hint'
import { useAtCompletions } from './hooks/use-at-completions'
import { useSlashCompletions } from './hooks/use-slash-completions'
import { useVoiceConversation } from './hooks/use-voice-conversation'
import { useVoiceRecorder } from './hooks/use-voice-recorder'
import { dragHasAttachments, droppedFileInlineRef, insertInlineRefsIntoEditor } from './inline-refs'
import { QueuePanel } from './queue-panel'
import {
  composerPlainText,
  placeCaretEnd,
  refChipElement,
  renderComposerContents,
  RICH_INPUT_SLOT
} from './rich-editor'
import { SkinSlashPopover } from './skin-slash-popover'
import { detectTrigger, extractClipboardImageBlobs, shouldSkipTriggerRefreshOnKeyUp, textBeforeCaret, type TriggerState } from './text-utils'
import { ComposerTriggerPopover } from './trigger-popover'
import type { ChatBarProps } from './types'
import { UrlDialog } from './url-dialog'
import { VoiceActivity, VoicePlaybackActivity } from './voice-activity'

const COMPOSER_STACK_BREAKPOINT_PX = 320

const COMPOSER_FADE_BACKGROUND =
  'linear-gradient(to bottom, transparent, color-mix(in srgb, var(--dt-background) 10%, transparent))'

interface QueueEditState {
  attachments: ComposerAttachment[]
  draft: string
  entryId: string
  sessionKey: string
}

const cloneAttachments = (attachments: ComposerAttachment[]) => attachments.map(a => ({ ...a }))

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
  onSubmit,
  onTranscribeAudio
}: ChatBarProps) {
  const aui = useAui()
  const draft = useAuiState(s => s.composer.text)
  const attachments = useStore($composerAttachments)
  const queuedPromptsBySession = useStore($queuedPromptsBySession)
  const scrolledUp = useStore($threadScrolledUp)
  const activeQueueSessionKey = queueSessionKey || sessionId || null

  const queuedPrompts = useMemo(
    () => (activeQueueSessionKey ? (queuedPromptsBySession[activeQueueSessionKey] ?? []) : []),
    [activeQueueSessionKey, queuedPromptsBySession]
  )

  const composerRef = useRef<HTMLFormElement | null>(null)
  const composerSurfaceRef = useRef<HTMLDivElement | null>(null)
  const editorRef = useRef<HTMLDivElement | null>(null)
  const draftRef = useRef(draft)
  const previousBusyRef = useRef(busy)
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
  const lastSpokenIdRef = useRef<string | null>(null)

  const narrow = useMediaQuery('(max-width: 30rem)')

  const at = useAtCompletions({ gateway: gateway ?? null, sessionId: sessionId ?? null, cwd: cwd ?? null })
  const slash = useSlashCompletions({ gateway: gateway ?? null })

  const stacked = expanded || narrow || tight
  const hasComposerPayload = draft.trim().length > 0 || attachments.length > 0
  const canSubmit = busy || hasComposerPayload
  const editingQueuedPrompt = queueEdit ? (queuedPrompts.find(entry => entry.id === queueEdit.entryId) ?? null) : null
  const busyAction = busy && hasComposerPayload ? 'queue' : 'stop'
  const showHelpHint = draft === '?'

  const placeholder = disabled ? 'Starting Hermes...' : 'Send follow-up'

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

  // Track expansion via cheap heuristics (newline or length threshold) instead
  // of reading editor.scrollHeight on every keystroke. scrollHeight forces a
  // synchronous layout flush — measured at 2.27 layouts per character typed
  // (see scripts/leak-typing.mjs). With ~30 chars before a typical wrap on
  // composer-default-width, this heuristic flips at roughly the right time
  // and the user only notices if they type far past the wrap boundary
  // without a newline; in that case the ResizeObserver below catches it via
  // a height delta and we still expand.
  useEffect(() => {
    if (!draft) {
      setExpanded(false)

      return
    }

    if (expanded) {
      return
    }

    if (draft.includes('\n') || draft.length > 60) {
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

  useResizeObserver(syncComposerMetrics, composerRef, composerSurfaceRef)

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

  const insertInlineRefs = (refs: string[]) => {
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

  const selectSkinSlashCommand = (command: string) => {
    draftRef.current = command
    aui.composer().setText(command)
    requestMainFocus()
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

    const pastedText = event.clipboardData.getData('text')

    if (!pastedText) {
      return
    }

    if (DATA_IMAGE_URL_RE.test(pastedText.trim())) {
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

  const handleEditorInput = (event: FormEvent<HTMLDivElement>) => {
    const editor = event.currentTarget

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

    const serialized = hermesDirectiveFormatter.serialize(item)
    const starter = serialized.endsWith(':')
    const text = starter || serialized.endsWith(' ') ? serialized : `${serialized} `
    const directive = !starter && serialized.match(/^@([^:]+):(.+)$/)

    const finish = () => {
      draftRef.current = composerPlainText(editor)
      aui.composer().setText(draftRef.current)
      requestMainFocus()
      starter ? window.setTimeout(refreshTrigger, 0) : closeTrigger()
    }

    const sel = window.getSelection()
    const range = sel?.rangeCount ? sel.getRangeAt(0) : null
    const node = range?.startContainer
    const offset = range?.startOffset ?? 0

    if (!sel || !range || node?.nodeType !== Node.TEXT_NODE || offset < trigger.tokenLength) {
      const current = composerPlainText(editor)
      renderComposerContents(editor, `${current.slice(0, Math.max(0, current.length - trigger.tokenLength))}${text}`)
      placeCaretEnd(editor)

      return finish()
    }

    const replaceRange = document.createRange()
    replaceRange.setStart(node, offset - trigger.tokenLength)
    replaceRange.setEnd(node, offset)
    replaceRange.deleteContents()

    if (directive) {
      const chip = refChipElement(directive[1], directive[2])
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
    if ((event.metaKey || event.ctrlKey) && !event.altKey && !event.shiftKey && event.key.toLowerCase() === 'k') {
      event.preventDefault()

      if (!busy) {
        void drainNextQueued()
      }

      return
    }

    if (trigger && triggerItems.length > 0) {
      if (event.key === 'ArrowDown') {
        event.preventDefault()
        setTriggerActive(idx => (idx + 1) % triggerItems.length)

        return
      }

      if (event.key === 'ArrowUp') {
        event.preventDefault()
        setTriggerActive(idx => (idx - 1 + triggerItems.length) % triggerItems.length)

        return
      }

      if (event.key === 'Enter' || event.key === 'Tab') {
        event.preventDefault()
        const item = triggerItems[triggerActive]

        if (item) {
          replaceTriggerWithChip(item)
        }

        return
      }

      if (event.key === 'Escape') {
        event.preventDefault()
        closeTrigger()

        return
      }
    }

    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()

      if (!busy && !hasComposerPayload && queuedPrompts.length > 0) {
        void drainNextQueued()

        return
      }

      submitDraft()
    }
  }

  const handleEditorKeyUp = (event: KeyboardEvent<HTMLDivElement>) => {
    // Arrow/Enter/Tab/Escape while the trigger menu is open are fully handled
    // in keydown and never edit text. Refreshing the trigger here would reset
    // the highlight to the top (breaking ArrowDown/ArrowUp) and re-open a menu
    // that Escape just closed, so skip it.
    if (shouldSkipTriggerRefreshOnKeyUp(event.key, trigger !== null)) {
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

    if (Array.from(event.dataTransfer.types || []).includes(HERMES_PATHS_MIME)) {
      const refs = candidates
        .map(candidate => droppedFileInlineRef(candidate, cwd))
        .filter((ref): ref is string => Boolean(ref))

      if (insertInlineRefs(refs)) {
        triggerHaptic('selection')
      }

      return
    }

    void Promise.resolve(onAttachDroppedItems(candidates)).then(attached => {
      if (attached) {
        triggerHaptic('selection')
        requestMainFocus()
      }
    })
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

    const refs = candidates
      .map(candidate => droppedFileInlineRef(candidate, cwd))
      .filter((ref): ref is string => Boolean(ref))

    if (!refs.length) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    resetDragState()

    if (insertInlineRefs(refs)) {
      triggerHaptic('selection')
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

        return true
      } finally {
        drainingQueueRef.current = false
      }
    },
    [activeQueueSessionKey, onSubmit, queuedPrompts]
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
    (id: string) => runDrain(entries => entries.find(e => e.id === id && id !== queueEdit?.entryId)),
    [queueEdit, runDrain]
  )

  const interruptAndSendNextQueued = useCallback(async () => {
    if (queuedPrompts.length === 0) {
      return false
    }

    await Promise.resolve(onCancel())

    return drainNextQueued()
  }, [drainNextQueued, onCancel, queuedPrompts.length])

  // Auto-drain on busy → false (turn settled).
  useEffect(() => {
    const wasBusy = previousBusyRef.current
    previousBusyRef.current = busy

    if (busy || !wasBusy || queuedPrompts.length === 0) {
      return
    }

    void drainNextQueued()
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
    if (queueEdit) {
      exitQueuedEdit('save')
    } else if (busy) {
      if (hasComposerPayload) {
        queueCurrentDraft()
      } else if (queuedPrompts.length > 0) {
        void interruptAndSendNextQueued()
      } else {
        triggerHaptic('cancel')
        void Promise.resolve(onCancel())
      }
    } else if (!hasComposerPayload && queuedPrompts.length > 0) {
      void drainNextQueued()
    } else if (draft.trim() || attachments.length > 0) {
      const submitted = draft
      triggerHaptic('submit')
      clearDraft()
      void onSubmit(submitted)
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
      state={state}
      voiceStatus={voiceStatus}
    />
  )

  const input = (
    <div className={cn('relative', stacked ? 'w-full' : 'min-w-(--composer-input-inline-min-width) flex-1')}>
      <div
        aria-label="Message"
        autoCorrect="off"
        autoCapitalize="off"
        className={cn(
          'min-h-(--composer-input-min-height) max-h-(--composer-input-max-height) overflow-y-auto bg-transparent pb-1 pr-1 pt-1 leading-normal text-foreground outline-none disabled:cursor-not-allowed',
          'empty:before:content-[attr(data-placeholder)] empty:before:text-muted-foreground/60',
          '**:data-ref-text:cursor-default',
          stacked && 'pl-3',
          stacked ? 'w-full' : 'min-w-(--composer-input-inline-min-width) flex-1'
        )}
        contentEditable={!disabled}
        data-placeholder={placeholder}
        data-slot={RICH_INPUT_SLOT}
        onBlur={() => window.setTimeout(closeTrigger, 80)}
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
      <ComposerPrimitive.Input asChild tabIndex={-1} unstable_focusOnScrollToBottom={false}>
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
          <SkinSlashPopover draft={draft} onSelect={selectSkinSlashCommand} />
          {activeQueueSessionKey && queuedPrompts.length > 0 && (
            <div className="relative z-6 mb-1 px-0.5">
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
                'relative z-4 isolate rounded-[inherit] border border-[color-mix(in_srgb,var(--dt-composer-ring)_calc(18%*var(--composer-ring-strength)),var(--dt-input))] shadow-composer transition-[border-color,box-shadow] duration-200 ease-out',
                COMPOSER_DROP_FADE_CLASS,
                'group-focus-within/composer:border-[color-mix(in_srgb,var(--dt-composer-ring)_calc(45%*var(--composer-ring-strength)),transparent)] group-focus-within/composer:shadow-composer-focus',
                'group-has-data-[state=open]/composer:border-t-transparent',
                'group-has-data-[state=open]/composer:shadow-[0_0.0625rem_0_0.0625rem_color-mix(in_srgb,var(--dt-composer-ring)_calc(35%*var(--composer-ring-strength)),transparent),0_0.5rem_1.5rem_color-mix(in_srgb,var(--shadow-ink)_6%,transparent)]',
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
                      Editing queued turn in composer
                    </div>
                    <div className="flex shrink-0 items-center gap-1">
                      <Button
                        className="h-6 rounded-md px-2 text-[0.68rem]"
                        onClick={() => exitQueuedEdit('cancel')}
                        type="button"
                        variant="ghost"
                      >
                        Cancel
                      </Button>
                      <Button
                        className="h-6 rounded-md px-2 text-[0.68rem]"
                        onClick={() => exitQueuedEdit('save')}
                        type="button"
                      >
                        Save
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
                      : 'grid-cols-[auto_1fr_auto] items-end gap-(--composer-control-gap) [grid-template-areas:"menu_input_controls"]'
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
      <div className="composer-fallback-surface relative isolate h-(--composer-fallback-height) w-full rounded-[inherit] border border-[color-mix(in_srgb,var(--dt-composer-ring)_calc(18%*var(--composer-ring-strength)),var(--dt-input))] shadow-composer">
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
