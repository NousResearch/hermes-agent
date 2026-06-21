import './liquid-glass-overrides.css'

import type { Unstable_TriggerAdapter, Unstable_TriggerItem } from '@assistant-ui/core'
import { ComposerPrimitive, useAui, useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import LiquidGlass from 'liquid-glass-react'
import {
  type ClipboardEvent,
  type FormEvent,
  type KeyboardEvent,
  type DragEvent as ReactDragEvent,
  useCallback,
  useEffect,
  useRef,
  useState
} from 'react'

import { formatRefValue, hermesDirectiveFormatter } from '@/components/assistant-ui/directive-text'
import { useMediaQuery } from '@/hooks/use-media-query'
import { chatMessageText } from '@/lib/chat-messages'
import { contextPath } from '@/lib/chat-runtime'
import { DATA_IMAGE_URL_RE, dataUrlToBlob } from '@/lib/embedded-images'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'
import { $composerAttachments, $composerDraft } from '@/store/composer'
import { $messages } from '@/store/session'
import { $threadScrolledUp } from '@/store/thread-scroll'

import { type DroppedFile, extractDroppedFiles, HERMES_PATHS_MIME } from '../hooks/use-composer-actions'

import { AttachmentList } from './attachments'
import { ContextMenu } from './context-menu'
import { ComposerControls } from './controls'
import { HelpHint } from './help-hint'
import { useAtCompletions } from './hooks/use-at-completions'
import { useSlashCompletions } from './hooks/use-slash-completions'
import { useVoiceConversation } from './hooks/use-voice-conversation'
import { useVoiceRecorder } from './hooks/use-voice-recorder'
import {
  composerPlainText,
  placeCaretEnd,
  refChipElement,
  renderComposerContents,
  RICH_INPUT_SLOT
} from './rich-editor'
import { SkinSlashPopover } from './skin-slash-popover'
import { ComposerTriggerPopover } from './trigger-popover'
import type { ChatBarProps } from './types'
import { UrlDialog } from './url-dialog'
import { VoiceActivity, VoicePlaybackActivity } from './voice-activity'

const COMPOSER_SHELL_CLASS =
  'group/composer absolute bottom-0 left-1/2 z-30 w-[min(var(--composer-width),calc(100%-2rem))] max-w-full -translate-x-1/2 rounded-2xl pt-2 pb-[var(--composer-shell-pad-block-end)]'

function extractClipboardImageBlobs(clipboard: DataTransfer): Blob[] {
  const blobs: Blob[] = []
  const seen = new Set<Blob>()

  const push = (blob: Blob | null) => {
    if (!blob || blob.size === 0 || seen.has(blob)) {
      return
    }

    seen.add(blob)
    blobs.push(blob)
  }

  if (clipboard.items?.length) {
    for (const item of clipboard.items) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        push(item.getAsFile())
      }
    }
  }

  if (clipboard.files?.length) {
    for (let i = 0; i < clipboard.files.length; i += 1) {
      const file = clipboard.files.item(i)

      if (file && file.type.startsWith('image/')) {
        push(file)
      }
    }
  }

  if (blobs.length > 0) {
    return blobs
  }

  const text = clipboard.getData('text/plain').trim()

  if (DATA_IMAGE_URL_RE.test(text)) {
    push(dataUrlToBlob(text))
  }

  if (blobs.length === 0) {
    const html = clipboard.getData('text/html')

    if (html) {
      const matches = html.matchAll(/<img\b[^>]*?\bsrc\s*=\s*["'](data:image\/[^"']+)["']/gi)

      for (const match of matches) {
        push(dataUrlToBlob(match[1]))
      }
    }
  }

  return blobs
}

const COMPOSER_STACK_BREAKPOINT_PX = 320

const COMPOSER_SCROLLED_DIM_CLASS =
  'opacity-30 group-hover/composer:opacity-100 group-focus-within/composer:opacity-100'

const COMPOSER_FROST_CLASS = cn(
  'pointer-events-none absolute inset-0 -z-10 rounded-[inherit]',
  'bg-[color-mix(in_srgb,var(--dt-card)_72%,transparent)]',
  'backdrop-blur-[0.75rem] backdrop-saturate-[1.12]',
  '[-webkit-backdrop-filter:blur(0.75rem)_saturate(1.12)]',
  'transition-[background-color,backdrop-filter,-webkit-backdrop-filter] duration-150 ease-out',
  'group-data-[thread-scrolled-up]/composer:bg-[color-mix(in_srgb,var(--dt-card)_48%,transparent)]',
  'group-focus-within/composer:bg-[var(--dt-card)]',
  'group-focus-within/composer:[backdrop-filter:none]',
  'group-focus-within/composer:[-webkit-backdrop-filter:none]'
)

const COMPOSER_GLASS = {
  fadeBackground: 'linear-gradient(to bottom, transparent, color-mix(in srgb, var(--dt-background) 10%, transparent))',
  liquidKey: ['standard', '0.950', '0.072', '0', '46', '0.00', '128'].join(':'),
  showLibraryRims: false,
  liquid: {
    aberrationIntensity: 0.95,
    blurAmount: 0.072,
    cornerRadius: 0,
    displacementScale: 46,
    elasticity: 0,
    mode: 'standard' as const,
    saturation: 128
  }
}

interface TriggerState {
  kind: '@' | '/'
  query: string
  tokenLength: number
}

const TRIGGER_RE = /(?:^|[\s])([@/])([^\s@/]*)$/

/** Caret-anchored text before the cursor, or null if the selection isn't a collapsed caret inside `editor`. */
function textBeforeCaret(editor: HTMLDivElement): string | null {
  const sel = window.getSelection()
  const range = sel?.rangeCount ? sel.getRangeAt(0) : null

  if (!range?.collapsed || !editor.contains(range.commonAncestorContainer)) {
    return null
  }

  const before = range.cloneRange()
  before.selectNodeContents(editor)
  before.setEnd(range.startContainer, range.startOffset)

  return before.toString()
}

function detectTrigger(textBefore: string): TriggerState | null {
  const match = TRIGGER_RE.exec(textBefore)

  if (!match) {
    return null
  }

  return { kind: match[1] as '@' | '/', query: match[2], tokenLength: 1 + match[2].length }
}

export function ChatBar({
  busy,
  cwd,
  disabled,
  focusKey,
  gateway,
  maxRecordingSeconds = 120,
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
  const scrolledUp = useStore($threadScrolledUp)

  const composerRef = useRef<HTMLFormElement | null>(null)
  const editorRef = useRef<HTMLDivElement | null>(null)
  const glassShellRef = useRef<HTMLDivElement | null>(null)
  const draftRef = useRef(draft)
  const urlInputRef = useRef<HTMLInputElement | null>(null)

  const [urlOpen, setUrlOpen] = useState(false)
  const [urlValue, setUrlValue] = useState('')
  const [expanded, setExpanded] = useState(false)
  const [voiceConversationActive, setVoiceConversationActive] = useState(false)
  const [tight, setTight] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const dragDepthRef = useRef(0)
  const lastSpokenIdRef = useRef<string | null>(null)

  const narrow = useMediaQuery('(max-width: 480px)')

  const at = useAtCompletions({ gateway: gateway ?? null, sessionId: sessionId ?? null, cwd: cwd ?? null })
  const slash = useSlashCompletions({ gateway: gateway ?? null })

  const stacked = expanded || narrow || tight
  const hasComposerPayload = draft.trim().length > 0 || attachments.length > 0
  const canSubmit = busy || hasComposerPayload
  const showHelpHint = draft === '?'

  const placeholder = disabled ? 'Starting 元话 Agent…' : 'Ask anything'

  const focusInput = () => window.requestAnimationFrame(() => editorRef.current?.focus({ preventScroll: true }))

  useEffect(() => {
    if (!disabled) {
      focusInput()
    }
  }, [disabled, focusKey])

  useEffect(() => {
    draftRef.current = draft
    $composerDraft.set(draft)

    const editor = editorRef.current

    if (editor && document.activeElement !== editor && composerPlainText(editor) !== draft) {
      renderComposerContents(editor, draft)
    }
  }, [draft])

  useEffect(
    () =>
      $composerDraft.subscribe(value => {
        if (value !== draftRef.current) {
          aui.composer().setText(value)
        }
      }),
    [aui]
  )

  useEffect(() => {
    if (urlOpen) {
      window.requestAnimationFrame(() => urlInputRef.current?.focus({ preventScroll: true }))
    }
  }, [urlOpen])

  useEffect(() => {
    if (!draft) {
      setExpanded(false)

      return
    }

    if (expanded) {
      return
    }

    const wraps = (editorRef.current?.scrollHeight ?? 0) > 56

    if (draft.includes('\n') || wraps) {
      setExpanded(true)
    }
  }, [draft, expanded])

  useEffect(() => {
    const el = composerRef.current

    if (!el) {
      return
    }

    const ro = new ResizeObserver(() => {
      const width = el.getBoundingClientRect().width

      if (width > 0) {
        setTight(width < COMPOSER_STACK_BREAKPOINT_PX)
      }
    })

    ro.observe(el)

    return () => ro.disconnect()
  }, [])

  const insertText = (text: string) => {
    const currentDraft = draftRef.current
    const sep = currentDraft && !currentDraft.endsWith('\n') ? '\n' : ''
    const nextDraft = `${currentDraft}${sep}${text}`

    draftRef.current = nextDraft
    aui.composer().setText(nextDraft)
    focusInput()
  }

  const insertInlineRefs = (refs: string[]) => {
    const editor = editorRef.current

    if (!refs.length || !editor) {
      return false
    }

    const inline = refs.join(' ')
    const selection = window.getSelection()

    const range =
      selection?.rangeCount && editor.contains(selection.getRangeAt(0).commonAncestorContainer)
        ? selection.getRangeAt(0)
        : null

    editor.focus({ preventScroll: true })

    if (range) {
      const beforeRange = range.cloneRange()
      beforeRange.selectNodeContents(editor)
      beforeRange.setEnd(range.startContainer, range.startOffset)
      const beforeContainer = document.createElement('div')
      beforeContainer.appendChild(beforeRange.cloneContents())

      const afterRange = range.cloneRange()
      afterRange.selectNodeContents(editor)
      afterRange.setStart(range.endContainer, range.endOffset)
      const afterContainer = document.createElement('div')
      afterContainer.appendChild(afterRange.cloneContents())

      const beforeText = composerPlainText(beforeContainer)
      const afterText = composerPlainText(afterContainer)
      const needsBeforeSpace = beforeText.length > 0 && !/\s$/.test(beforeText)
      const needsAfterSpace = afterText.length === 0 || !/^\s/.test(afterText)
      range.deleteContents()
      const fragment = document.createDocumentFragment()

      if (needsBeforeSpace) {
        fragment.appendChild(document.createTextNode(' '))
      }

      refs.forEach((ref, index) => {
        const match = ref.match(/^@([^:]+):(.+)$/)
        fragment.appendChild(match ? refChipElement(match[1], match[2]) : document.createTextNode(ref))

        if (index < refs.length - 1) {
          fragment.appendChild(document.createTextNode(' '))
        }
      })

      const trailingSpace = needsAfterSpace ? document.createTextNode(' ') : null

      if (trailingSpace) {
        fragment.appendChild(trailingSpace)
      }

      range.insertNode(fragment)

      const nextRange = document.createRange()

      if (trailingSpace) {
        nextRange.setStart(trailingSpace, trailingSpace.length)
      } else {
        nextRange.setStartAfter(fragment.lastChild || range.startContainer)
      }

      nextRange.collapse(true)
      selection?.removeAllRanges()
      selection?.addRange(nextRange)
    } else {
      const current = composerPlainText(editor)
      renderComposerContents(editor, `${current}${current && !/\s$/.test(current) ? ' ' : ''}${inline} `)
      placeCaretEnd(editor)
    }

    const nextDraft = composerPlainText(editor)
    draftRef.current = nextDraft
    aui.composer().setText(nextDraft)

    return true
  }

  const droppedFileInlineRef = (candidate: DroppedFile) => {
    if (!candidate.path) {
      return null
    }

    const rel = contextPath(candidate.path, cwd || '')

    if (candidate.line) {
      const { line, lineEnd } = candidate
      const range = lineEnd && lineEnd > line ? `${line}-${lineEnd}` : `${line}`

      return `@line:${formatRefValue(`${rel}:${range}`)}`
    }

    const kind = candidate.isDirectory ? 'folder' : 'file'

    return `@${kind}:${formatRefValue(rel)}`
  }

  const selectSkinSlashCommand = (command: string) => {
    draftRef.current = command
    aui.composer().setText(command)
    focusInput()
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

    const before = textBeforeCaret(editor)
    const detected = detectTrigger(before ?? composerPlainText(editor))

    setTrigger(detected)
    setTriggerActive(0)
  }, [])

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
      submitDraft()
    }
  }

  const handleEditorKeyUp = () => {
    window.setTimeout(refreshTrigger, 0)
  }

  const dragHasAttachments = (transfer: DataTransfer | null) => {
    if (!transfer) {
      return false
    }

    if (Array.from(transfer.types || []).includes(HERMES_PATHS_MIME)) {
      return true
    }

    if (Array.from(transfer.types || []).includes('Files')) {
      return true
    }

    return Array.from(transfer.items || []).some(item => item.kind === 'file')
  }

  const resetDragState = () => {
    dragDepthRef.current = 0
    setDragActive(false)
  }

  const handleDragEnter = (event: ReactDragEvent<HTMLFormElement>) => {
    if (!onAttachDroppedItems || !dragHasAttachments(event.dataTransfer)) {
      return
    }

    event.preventDefault()
    dragDepthRef.current += 1

    if (!dragActive) {
      setDragActive(true)
    }
  }

  const handleDragOver = (event: ReactDragEvent<HTMLFormElement>) => {
    if (!onAttachDroppedItems || !dragHasAttachments(event.dataTransfer)) {
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
      const refs = candidates.map(droppedFileInlineRef).filter((ref): ref is string => Boolean(ref))

      if (insertInlineRefs(refs)) {
        triggerHaptic('selection')
      }

      return
    }

    void Promise.resolve(onAttachDroppedItems(candidates)).then(attached => {
      if (attached) {
        triggerHaptic('selection')
        focusInput()
      }
    })
  }

  const handleInputDragOver = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!dragHasAttachments(event.dataTransfer)) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    event.dataTransfer.dropEffect = 'copy'
  }

  const handleInputDrop = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!dragHasAttachments(event.dataTransfer)) {
      return
    }

    const candidates = extractDroppedFiles(event.dataTransfer)
    const refs = candidates.map(droppedFileInlineRef).filter((ref): ref is string => Boolean(ref))

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

  const clearDraft = () => {
    aui.composer().setText('')
    draftRef.current = ''

    if (editorRef.current) {
      editorRef.current.replaceChildren()
    }
  }

  const submitDraft = () => {
    if (busy) {
      triggerHaptic('cancel')
      onCancel()
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
        onInput={handleEditorInput}
        onKeyDown={handleEditorKeyDown}
        onKeyUp={handleEditorKeyUp}
        onMouseUp={refreshTrigger}
        onPaste={handlePaste}
        ref={editorRef}
        role="textbox"
        suppressContentEditableWarning
      />
      <ComposerPrimitive.Input className="sr-only" tabIndex={-1} unstable_focusOnScrollToBottom={false} />
    </div>
  )

  return (
    <>
      <ComposerPrimitive.Unstable_TriggerPopoverRoot>
        <ComposerPrimitive.Root
          className={COMPOSER_SHELL_CLASS}
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
          <div
            className="pointer-events-none absolute inset-0 rounded-[inherit]"
            style={{ background: COMPOSER_GLASS.fadeBackground }}
          />
          <div className="relative w-full rounded-[inherit]">
            <div
              className={cn(
                'composer-liquid-shell-wrap absolute -inset-px isolate overflow-hidden rounded-[calc(var(--radius-2xl)+1px)] transition-opacity duration-200 ease-out',
                scrolledUp ? COMPOSER_SCROLLED_DIM_CLASS : 'opacity-100'
              )}
              data-glass-frame="true"
              data-show-library-rims={COMPOSER_GLASS.showLibraryRims ? 'true' : undefined}
              data-slot="composer-liquid-shell-wrap"
              ref={glassShellRef}
            >
              <LiquidGlass
                aberrationIntensity={COMPOSER_GLASS.liquid.aberrationIntensity}
                blurAmount={COMPOSER_GLASS.liquid.blurAmount}
                className="composer-liquid-shell pointer-events-none absolute inset-0 h-full w-full"
                cornerRadius={COMPOSER_GLASS.liquid.cornerRadius}
                displacementScale={COMPOSER_GLASS.liquid.displacementScale}
                elasticity={COMPOSER_GLASS.liquid.elasticity}
                key={COMPOSER_GLASS.liquidKey}
                mode={COMPOSER_GLASS.liquid.mode}
                mouseContainer={composerRef}
                padding="0"
                saturation={COMPOSER_GLASS.liquid.saturation}
                style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
              >
                <span className="block h-full w-full" />
              </LiquidGlass>
            </div>
            <div
              className={cn(
                'relative z-4 isolate rounded-[inherit] border border-[color-mix(in_srgb,var(--dt-midground)_18%,var(--dt-input))] shadow-composer transition-[border-color,box-shadow] duration-200 ease-out',
                'group-focus-within/composer:border-ring/45 group-focus-within/composer:shadow-composer-focus',
                'group-has-data-[state=open]/composer:border-t-transparent',
                'group-has-data-[state=open]/composer:shadow-[0_0.0625rem_0_0.0625rem_color-mix(in_srgb,var(--dt-ring)_35%,transparent),0_0.5rem_1.5rem_color-mix(in_srgb,var(--shadow-ink)_6%,transparent)]',
                dragActive && 'border-midground/70 shadow-composer-focus ring-2 ring-midground/40'
              )}
              data-slot="composer-surface"
            >
              <div aria-hidden className={COMPOSER_FROST_CLASS} />
              {dragActive && (
                <div
                  aria-hidden
                  className="pointer-events-none absolute inset-0 z-3 flex items-center justify-center bg-midground/10 text-sm font-semibold uppercase tracking-[0.18em] text-midground backdrop-blur-[1px]"
                >
                  Drop files to attach
                </div>
              )}
              <div
                className={cn(
                  'relative z-1 flex min-h-0 w-full flex-col gap-(--composer-row-gap) overflow-hidden rounded-[inherit] px-(--composer-surface-pad-x) py-(--composer-surface-pad-y) transition-opacity duration-200 ease-out',
                  scrolledUp ? COMPOSER_SCROLLED_DIM_CLASS : 'opacity-100'
                )}
                data-slot="composer-fade"
              >
                <VoiceActivity state={voiceActivityState} />
                <VoicePlaybackActivity />
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
      className={cn(COMPOSER_SHELL_CLASS, 'bg-linear-to-b from-transparent to-background/55')}
      data-slot="composer-root"
    >
      <div className="composer-fallback-surface relative isolate h-(--composer-fallback-height) w-full rounded-[inherit] border border-[color-mix(in_srgb,var(--dt-midground)_18%,var(--dt-input))] shadow-composer">
        <div aria-hidden className={COMPOSER_FROST_CLASS} />
      </div>
    </div>
  )
}
