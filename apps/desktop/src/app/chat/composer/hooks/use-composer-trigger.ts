import type { Unstable_TriggerAdapter, Unstable_TriggerItem } from '@assistant-ui/core'
import { type MutableRefObject, type RefObject, useCallback, useEffect, useRef, useState } from 'react'

import { hermesDirectiveFormatter } from '@/components/assistant-ui/directive-text'
import { desktopSlashCommandTakesArgs } from '@/lib/desktop-slash-commands'

import { COMPLETION_ACTIONS, slashArgStage, slashChipKindForItem, slashCommandToken } from '../composer-utils'
import {
  composerPlainText,
  placeCaretEnd,
  refChipElement,
  renderComposerContents,
  slashChipElement
} from '../rich-editor'
import { detectTrigger, textBeforeCaret, type TriggerState } from '../text-utils'

interface CompletionSource {
  adapter: Unstable_TriggerAdapter | null
  loading: boolean
}

interface UseComposerTriggerOptions {
  at: CompletionSource
  draftRef: MutableRefObject<string>
  editorRef: RefObject<HTMLDivElement | null>
  requestMainFocus: () => void
  setComposerText: (text: string) => void
  slash: CompletionSource
}

/**
 * Trigger / completion engine: `@`/`/` detection against the live editor, the
 * adapter-driven item list, the open popover's selection state, and the chip
 * insertion that commits a pick back into the contentEditable. Owns the trigger
 * state; ChatBar threads its editor refs in and consumes the returned API from
 * the input/keydown/keyup paths + the popover render. `triggerKeyConsumedRef` is
 * exposed so keydown can mark a navigation/control key as handled and the
 * subsequent keyup skips its refresh.
 */
export function useComposerTrigger({
  at,
  draftRef,
  editorRef,
  requestMainFocus,
  setComposerText,
  slash
}: UseComposerTriggerOptions) {
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
    const found = detectTrigger(before ?? composerPlainText(editor))

    // The arg-stage popover is only useful for commands with an options screen.
    // For a no-arg command it would dead-end on "No matches", so drop it — the
    // directive is already complete.
    const detected =
      found?.kind === '/' && slashArgStage(found.query) && !desktopSlashCommandTakesArgs(slashCommandToken(found.query))
        ? null
        : found

    setTrigger(detected)

    // Only reset the highlight when the trigger actually changed (opened, or
    // the query/kind differs). Re-detecting the *same* trigger — e.g. on a
    // caret move (mouseup) or a stray refresh — must preserve the user's
    // current selection instead of snapping back to the first item.
    if (detected?.kind !== trigger?.kind || detected?.query !== trigger?.query) {
      setTriggerActive(0)
    }
  }, [editorRef, trigger])

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

  // Suppress the "No matches" empty state once a slash command is past its name:
  // a no-arg command has nothing to offer, and a fully-typed arg commits on
  // Space/Tab — neither should dead-end on a popover.
  const argStageEmpty = trigger?.kind === '/' && slashArgStage(trigger.query) && !triggerLoading && !triggerItems.length

  const closeTrigger = () => {
    setTrigger(null)
    setTriggerItems([])
    setTriggerActive(0)
  }

  useEffect(() => {
    setTriggerActive(idx => Math.min(idx, Math.max(0, triggerItems.length - 1)))
  }, [triggerItems.length])

  // Commit the literally-typed `/command arg` as a directive chip — used when
  // the completion list is empty because the arg is already fully typed (the
  // backend completer drops exact matches). Reuses the chip path via a
  // synthetic item whose serialized form is the verbatim text.
  const commitTypedSlashDirective = () => {
    if (trigger?.kind !== '/') {
      return
    }

    const text = `/${trigger.query.trimEnd()}`

    replaceTriggerWithChip({
      id: text,
      type: 'slash',
      label: text.slice(1),
      metadata: {
        command: slashCommandToken(trigger.query),
        display: text,
        meta: '',
        group: '',
        action: '',
        rawText: text
      }
    })
  }

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
      setComposerText(draftRef.current)
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

    const expandsToArgs = trigger.kind === '/' && !serialized.includes(' ') && desktopSlashCommandTakesArgs(command)

    const text = starter || serialized.endsWith(' ') ? serialized : `${serialized} `
    const directive = !starter && serialized.match(/^@([^:]+):(.+)$/)
    // No pill while expanding — the bare command stays plain text until an arg
    // is picked, at which point a single pill is emitted for the full command.
    const slashKind = !expandsToArgs && trigger.kind === '/' ? slashChipKindForItem(item) : null
    const keepTriggerOpen = starter || expandsToArgs

    const finish = () => {
      draftRef.current = composerPlainText(editor)
      setComposerText(draftRef.current)
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

  return {
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
  }
}
