import type { Unstable_TriggerAdapter, Unstable_TriggerItem } from '@assistant-ui/core'
import { ComposerPrimitive, useAui, useAuiState } from '@assistant-ui/react'
import {
  type ClipboardEvent,
  type FC,
  type FocusEvent,
  type FormEvent,
  type KeyboardEvent,
  type DragEvent as ReactDragEvent,
  useCallback,
  useEffect,
  useRef,
  useState
} from 'react'

import { COMPOSER_DROP_ACTIVE_CLASS, COMPOSER_DROP_FADE_CLASS } from '@/app/chat/composer/drop-affordance'
import {
  type ComposerInsertMode,
  focusComposerInput,
  markActiveComposer,
  onComposerFocusRequest,
  onComposerInsertRequest
} from '@/app/chat/composer/focus'
import { useAtCompletions } from '@/app/chat/composer/hooks/use-at-completions'
import { useSlashCompletions } from '@/app/chat/composer/hooks/use-slash-completions'
import {
  dragHasAttachments,
  droppedFileInlineRefs,
  type InlineRefInput,
  insertInlineRefsIntoEditor
} from '@/app/chat/composer/inline-refs'
import {
  composerPlainText,
  placeCaretEnd,
  refChipElement,
  renderComposerContents,
  RICH_INPUT_SLOT
} from '@/app/chat/composer/rich-editor'
import { detectTrigger, textBeforeCaret, type TriggerState } from '@/app/chat/composer/text-utils'
import { ComposerTriggerPopover } from '@/app/chat/composer/trigger-popover'
import {
  extractDroppedFiles,
  HERMES_PATHS_MIME,
  isImagePath,
  partitionDroppedFiles
} from '@/app/chat/hooks/use-composer-actions'
import { uploadComposerAttachment } from '@/app/session/hooks/use-prompt-actions'
import { hermesDirectiveFormatter } from '@/components/assistant-ui/directive-text'
import {
  StickyHumanMessageContainer,
  StopGlyph,
  USER_ACTION_ICON_BUTTON_CLASS,
  USER_ACTION_ICON_SIZE,
  USER_BUBBLE_BASE_CLASS
} from '@/components/assistant-ui/thread/user-message'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import type { HermesGateway } from '@/hermes'
import { useI18n } from '@/i18n'
import { attachmentDisplayText, attachmentId, pathLabel } from '@/lib/chat-runtime'
import { DATA_IMAGE_URL_RE } from '@/lib/embedded-images'
import { triggerHaptic } from '@/lib/haptics'
import { Loader2Icon } from '@/lib/icons'
import { cn } from '@/lib/utils'
import type { ComposerAttachment } from '@/store/composer'
import { notifyError } from '@/store/notifications'
import { $connection } from '@/store/session'
import { notifyThreadEditClose } from '@/store/thread-scroll'

interface UserEditComposerProps {
  cwd: string | null
  gateway: HermesGateway | null
  sessionId: string | null
}

export const UserEditComposer: FC<UserEditComposerProps> = ({ cwd, gateway, sessionId }) => {
  const { t } = useI18n()
  const copy = t.assistant.thread
  const aui = useAui()
  const draft = useAuiState(s => s.composer.text)
  const rootRef = useRef<HTMLDivElement | null>(null)
  const editorRef = useRef<HTMLDivElement | null>(null)
  const draftRef = useRef(draft)
  const dragDepthRef = useRef(0)
  const [dragActive, setDragActive] = useState(false)
  const [trigger, setTrigger] = useState<TriggerState | null>(null)
  const [triggerActive, setTriggerActive] = useState(0)
  const [triggerItems, setTriggerItems] = useState<readonly Unstable_TriggerItem[]>([])
  // See index.tsx: set in keydown when the open popover consumes a nav/control
  // key so the matching keyup skips refreshTrigger (timing-immune vs reading
  // `trigger`, which keyup sees as already-null after Escape).
  const triggerKeyConsumedRef = useRef(false)
  const [triggerPlacement, setTriggerPlacement] = useState<'bottom' | 'top'>('top')
  const [focusRequestId, setFocusRequestId] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  // True while OS-drop files are being staged/uploaded into the session. Blocks
  // submit and shows a spinner so confirming the edit can't race the async
  // upload and drop the gateway-side ref before it lands in the draft.
  const [staging, setStaging] = useState(false)
  const expanded = draft.includes('\n')
  const canSubmit = draft.trim().length > 0
  const at = useAtCompletions({ cwd, gateway, sessionId })
  const slash = useSlashCompletions({ gateway })

  useEffect(() => () => notifyThreadEditClose(), [])

  const focusEditor = useCallback(() => {
    const editor = editorRef.current

    focusComposerInput(editor)

    if (editor) {
      placeCaretEnd(editor)
    }

    markActiveComposer('edit')
  }, [])

  const requestEditFocus = useCallback(() => {
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
    draftRef.current = draft

    const editor = editorRef.current

    if (
      editor &&
      (editor.childNodes.length === 0 || (document.activeElement !== editor && composerPlainText(editor) !== draft))
    ) {
      renderComposerContents(editor, draft)

      if (document.activeElement === editor) {
        placeCaretEnd(editor)
      }
    }
  }, [draft])

  useEffect(() => {
    focusEditor()
  }, [focusEditor, focusRequestId])

  useEffect(() => {
    const offFocus = onComposerFocusRequest(target => {
      if (target === 'edit') {
        setFocusRequestId(id => id + 1)
      }
    })

    const offInsert = onComposerInsertRequest(({ mode, target, text }) => {
      if (target === 'edit') {
        appendExternalText(text, mode)
      }
    })

    return () => {
      offFocus()
      offInsert()
    }
  }, [appendExternalText])

  const syncDraftFromEditor = useCallback(
    (editor: HTMLDivElement) => {
      const nextDraft = composerPlainText(editor)

      if (nextDraft !== draftRef.current) {
        draftRef.current = nextDraft
        aui.composer().setText(nextDraft)
      }

      return nextDraft
    },
    [aui]
  )

  const refreshTrigger = useCallback(() => {
    const editor = editorRef.current

    if (!editor) {
      return
    }

    const before = textBeforeCaret(editor)
    const detected = detectTrigger(before ?? composerPlainText(editor))

    if (detected) {
      const rect = editor.getBoundingClientRect()
      const spaceAbove = rect.top
      const spaceBelow = window.innerHeight - rect.bottom

      setTriggerPlacement(spaceAbove < 220 && spaceBelow > spaceAbove ? 'bottom' : 'top')
    }

    setTrigger(detected)

    // Only reset the highlight when the trigger actually changed (opened, or
    // the query/kind differs). Re-detecting the *same* trigger — e.g. on a
    // caret move (mouseup) or a stray refresh — must preserve the user's
    // current selection instead of snapping back to the first item.
    if (detected?.kind !== trigger?.kind || detected?.query !== trigger?.query) {
      setTriggerActive(0)
    }
  }, [trigger])

  const closeTrigger = useCallback(() => {
    setTrigger(null)
    setTriggerItems([])
    setTriggerActive(0)
  }, [])

  const triggerAdapter: Unstable_TriggerAdapter | null =
    trigger?.kind === '@' ? at.adapter : trigger?.kind === '/' ? slash.adapter : null

  useEffect(() => {
    if (!trigger || !triggerAdapter?.search) {
      setTriggerItems([])

      return
    }

    setTriggerItems(triggerAdapter.search(trigger.query))
  }, [trigger, triggerAdapter])

  useEffect(() => {
    setTriggerActive(idx => Math.min(idx, Math.max(0, triggerItems.length - 1)))
  }, [triggerItems.length])

  const triggerLoading = trigger?.kind === '@' ? at.loading : trigger?.kind === '/' ? slash.loading : false

  const replaceTriggerWithChip = useCallback(
    (item: Unstable_TriggerItem) => {
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
        requestEditFocus()
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
    },
    [aui, closeTrigger, refreshTrigger, requestEditFocus, trigger]
  )

  const insertRefStrings = useCallback(
    (refs: InlineRefInput[]) => {
      const editor = editorRef.current

      if (!editor || refs.length === 0) {
        return false
      }

      const nextDraft = insertInlineRefsIntoEditor(editor, refs)

      if (nextDraft === null) {
        return false
      }

      draftRef.current = nextDraft
      aui.composer().setText(nextDraft)
      requestEditFocus()

      return true
    },
    [aui, requestEditFocus]
  )

  const insertDroppedRefs = useCallback(
    (candidates: ReturnType<typeof extractDroppedFiles>) => insertRefStrings(droppedFileInlineRefs(candidates, cwd)),
    [cwd, insertRefStrings]
  )

  // OS/Finder drops carry an absolute path on THIS machine — the gateway can't
  // read it in remote mode, and an image needs its bytes uploaded for vision.
  // Stage each through the same file.attach/image.attach_bytes pipeline the main
  // composer uses, then insert the *gateway-side* ref the agent can resolve —
  // never the raw local path (the MahmoudR remote-attach bug, which the main
  // composer fixes but this edit composer used to reproduce).
  const uploadOsDropRefs = useCallback(
    async (osDrops: ReturnType<typeof extractDroppedFiles>): Promise<InlineRefInput[]> => {
      if (!gateway || !sessionId) {
        // No session to stage into — best-effort inline refs (matches old path).
        return droppedFileInlineRefs(osDrops, cwd)
      }

      const remote = $connection.get()?.mode === 'remote'

      const requestGateway = <T,>(method: string, params?: Record<string, unknown>) =>
        gateway.request<T>(method, params)

      const refs: InlineRefInput[] = []

      for (const candidate of osDrops) {
        const path = candidate.path || ''

        if (!path) {
          continue
        }

        const kind: ComposerAttachment['kind'] =
          candidate.file?.type.startsWith('image/') || isImagePath(candidate.file?.name || path) ? 'image' : 'file'

        try {
          const uploaded = await uploadComposerAttachment(
            { detail: path, id: attachmentId(kind, path), kind, label: pathLabel(path), path },
            { remote, requestGateway, sessionId }
          )

          const ref = attachmentDisplayText(uploaded)

          if (ref) {
            refs.push(ref)
          }
        } catch (err) {
          notifyError(err, t.desktop.dropFiles)
        }
      }

      return refs
    },
    [cwd, gateway, sessionId, t.desktop.dropFiles]
  )

  const resetDragState = useCallback(() => {
    dragDepthRef.current = 0
    setDragActive(false)
  }, [])

  const handleDragEnter = (event: ReactDragEvent<HTMLElement>) => {
    if (!dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME)) {
      return
    }

    event.preventDefault()
    dragDepthRef.current += 1

    if (!dragActive) {
      setDragActive(true)
    }
  }

  const handleDragOver = (event: ReactDragEvent<HTMLElement>) => {
    if (!dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME)) {
      return
    }

    event.preventDefault()
    event.dataTransfer.dropEffect = 'copy'
  }

  const handleDragLeave = (event: ReactDragEvent<HTMLElement>) => {
    event.preventDefault()
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1)

    if (dragDepthRef.current === 0) {
      setDragActive(false)
    }
  }

  const handleDrop = (event: ReactDragEvent<HTMLElement>) => {
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

    // In-app drags (project tree / gutter) are workspace-relative paths that
    // resolve on the gateway as-is, so they stay inline refs. OS drops need to
    // be staged + uploaded first, then their gateway-side ref is inserted.
    const { inAppRefs, osDrops } = partitionDroppedFiles(candidates)

    if (insertDroppedRefs(inAppRefs)) {
      triggerHaptic('selection')
    }

    if (osDrops.length) {
      setStaging(true)
      void uploadOsDropRefs(osDrops)
        .then(refs => {
          if (insertRefStrings(refs)) {
            triggerHaptic('selection')
          }
        })
        .finally(() => setStaging(false))
    }
  }

  const handleInput = (event: FormEvent<HTMLDivElement>) => {
    const editor = event.currentTarget

    if (editor.childNodes.length === 1 && editor.firstChild?.nodeName === 'BR') {
      editor.replaceChildren()
    }

    syncDraftFromEditor(editor)
    window.setTimeout(refreshTrigger, 0)
  }

  const handlePaste = (event: ClipboardEvent<HTMLDivElement>) => {
    const pastedText = event.clipboardData.getData('text')

    if (!pastedText || DATA_IMAGE_URL_RE.test(pastedText.trim())) {
      event.preventDefault()

      return
    }

    event.preventDefault()
    document.execCommand('insertText', false, pastedText)
    syncDraftFromEditor(event.currentTarget)
  }

  const submitEdit = (editor: HTMLDivElement) => {
    const nextDraft = syncDraftFromEditor(editor)

    if (submitting || staging || !nextDraft.trim()) {
      return
    }

    setSubmitting(true)
    aui.composer().send()
  }

  const handleEditBlur = useCallback(
    (event: FocusEvent<HTMLDivElement>) => {
      const nextTarget = event.relatedTarget

      if (nextTarget instanceof Node && event.currentTarget.contains(nextTarget)) {
        return
      }

      window.setTimeout(() => {
        const root = rootRef.current
        const active = document.activeElement

        if (submitting || (root && active && root.contains(active))) {
          return
        }

        closeTrigger()
        aui.composer().cancel()
      }, 80)
    },
    [aui, closeTrigger, submitting]
  )

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
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

    if (event.key === 'Escape') {
      event.preventDefault()
      aui.composer().cancel()

      return
    }

    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submitEdit(event.currentTarget)
    }
  }

  const handleKeyUp = () => {
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

  return (
    <ComposerPrimitive.Root className="contents" data-slot="aui_edit-composer-root">
      <StickyHumanMessageContainer>
        <div
          className="composer-human-message-container human-execution-message-top relative flex w-full items-start rounded-md bg-(--ui-chat-surface-background)"
          onBlur={handleEditBlur}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          ref={rootRef}
        >
          {trigger && (
            <ComposerTriggerPopover
              activeIndex={triggerActive}
              items={triggerItems}
              kind={trigger.kind}
              loading={triggerLoading}
              onHover={setTriggerActive}
              onPick={replaceTriggerWithChip}
              placement={triggerPlacement}
            />
          )}
          <div
            className={cn(
              USER_BUBBLE_BASE_CLASS,
              'ui-prompt-input__container relative border-(--ui-stroke-secondary) data-[expanded=true]:min-h-20',
              COMPOSER_DROP_FADE_CLASS,
              dragActive && COMPOSER_DROP_ACTIVE_CLASS
            )}
            data-expanded={expanded ? 'true' : undefined}
          >
            <div
              aria-label={copy.editMessage}
              autoCapitalize="off"
              autoCorrect="off"
              className={cn(
                'ui-prompt-input-editor__input max-h-48 w-full resize-none bg-transparent p-0 pr-7 text-[length:var(--conversation-text-font-size)] text-foreground/95 outline-none',
                'empty:before:content-[attr(data-placeholder)] empty:before:text-muted-foreground/60',
                '**:data-ref-text:cursor-default',
                expanded ? 'min-h-16' : 'min-h-[1.25rem]'
              )}
              contentEditable
              data-placeholder={copy.editMessage}
              data-slot={RICH_INPUT_SLOT}
              onBlur={() => window.setTimeout(closeTrigger, 80)}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onFocus={() => markActiveComposer('edit')}
              onInput={handleInput}
              onKeyDown={handleKeyDown}
              onKeyUp={handleKeyUp}
              onMouseUp={refreshTrigger}
              onPaste={handlePaste}
              ref={editorRef}
              role="textbox"
              spellCheck={false}
              suppressContentEditableWarning
            />
            <ComposerPrimitive.Input
              asChild
              className="sr-only"
              submitMode="ctrlEnter"
              tabIndex={-1}
              unstable_focusOnScrollToBottom={false}
            >
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
            {staging && (
              <span
                className="pointer-events-none absolute bottom-2 left-2 inline-flex items-center gap-1 rounded-full bg-background/80 px-1.5 py-0.5 text-[0.62rem] text-muted-foreground backdrop-blur-[1px]"
                data-slot="aui_edit-staging"
              >
                <Loader2Icon className="size-3 animate-spin" />
                {copy.attachingFile}
              </span>
            )}
            <Tip label={copy.sendEdited} side="top">
              <button
                aria-label={copy.sendEdited}
                className={cn('absolute right-2 bottom-2 size-5', USER_ACTION_ICON_BUTTON_CLASS)}
                disabled={!canSubmit || submitting || staging}
                onClick={() => {
                  const editor = editorRef.current

                  if (editor) {
                    submitEdit(editor)
                  }
                }}
                type="button"
              >
                {submitting ? StopGlyph : <Codicon name="arrow-up" size={USER_ACTION_ICON_SIZE} />}
              </button>
            </Tip>
          </div>
        </div>
      </StickyHumanMessageContainer>
    </ComposerPrimitive.Root>
  )
}
