import type { Unstable_TriggerAdapter, Unstable_TriggerItem } from '@assistant-ui/core'
import { type MutableRefObject, type RefObject, useCallback, useEffect, useRef, useState } from 'react'

import { hermesDirectiveFormatter } from '@/components/assistant-ui/directive-text'
import { desktopSlashCommandArgumentMode } from '@/lib/desktop-slash-commands'

import {
  COMPLETION_ACTIONS,
  isSkillItem,
  slashArgStage,
  slashChipKindForItem,
  slashCommandToken
} from '../composer-utils'
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
  // The list highlights its first row on open, which is a suggestion rather
  // than a choice. This records that the user moved the highlight themselves,
  // which is what lets Enter accept a completion in a free-text argument stage
  // without stealing prose from everyone who never touched the arrows.
  const [triggerActiveExplicit, setTriggerActiveExplicit] = useState(false)
  const [triggerItems, setTriggerItems] = useState<readonly Unstable_TriggerItem[]>([])
  // Set synchronously in keydown when the open trigger popover consumes a
  // navigation/control key (Arrow/Enter/Tab/Escape). The subsequent keyup must
  // NOT run refreshTrigger for that keypress: it never edits text, and for
  // Escape the keydown has already set trigger=null, so a keyup refresh would
  // re-detect the still-present `/` and instantly reopen the menu. A ref is
  // used instead of reading `trigger` in keyup because by keyup time React has
  // re-rendered and the handler closure sees the post-keydown state.
  const triggerKeyConsumedRef = useRef(false)

  const resetTriggerActive = useCallback(() => {
    setTriggerActive(0)
    setTriggerActiveExplicit(false)
  }, [])

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
        resetTriggerActive()
      }

      return
    }

    const before = textBeforeCaret(editor)
    const found = detectTrigger(before ?? composerPlainText(editor))

    // A text-only command has no completion screen once its prose begins. Mixed
    // commands such as /goal stay live so their finite subcommands can still be
    // suggested, while arbitrary goal text remains valid.
    const argumentMode =
      found?.kind === '/' && slashArgStage(found.query)
        ? desktopSlashCommandArgumentMode(slashCommandToken(found.query))
        : null

    const detected =
      found?.kind === '/' && slashArgStage(found.query) && argumentMode !== 'options' && argumentMode !== 'mixed'
        ? null
        : found

    setTrigger(detected)

    // Only reset the highlight when the trigger actually changed (opened, or
    // the query/kind differs). Re-detecting the *same* trigger — e.g. on a
    // caret move (mouseup) or a stray refresh — must preserve the user's
    // current selection instead of snapping back to the first item.
    if (detected?.kind !== trigger?.kind || detected?.query !== trigger?.query) {
      resetTriggerActive()
    }
  }, [editorRef, resetTriggerActive, trigger])

  const triggerAdapter: Unstable_TriggerAdapter | null =
    trigger?.kind === '@' ? at.adapter : trigger?.kind === '/' ? slash.adapter : null

  useEffect(() => {
    if (!trigger || !triggerAdapter?.search) {
      setTriggerItems([])

      return
    }

    const items = triggerAdapter.search(trigger.query)

    // Mid-message only offers SKILLS. A built-in like `/model` or `/new` acts
    // on the app, so it's meaningless as a reference inside prose — only a
    // skill reads as "handle this part with X". Filtering here rather than in
    // the fetcher keeps one completion source for both shapes.
    setTriggerItems(trigger.inline ? items.filter(isSkillItem) : items)
  }, [trigger, triggerAdapter])

  const triggerLoading = trigger?.kind === '@' ? at.loading : trigger?.kind === '/' ? slash.loading : false

  // Suppress the "No matches" empty state once a slash command is past its name:
  // a no-arg command has nothing to offer, and a fully-typed arg commits on
  // Space/Tab — neither should dead-end on a popover.
  const argStageEmpty = trigger?.kind === '/' && slashArgStage(trigger.query) && !triggerLoading && !triggerItems.length

  const slashArgumentMode =
    trigger?.kind === '/' && slashArgStage(trigger.query)
      ? desktopSlashCommandArgumentMode(slashCommandToken(trigger.query))
      : null

  const slashFreeTextArgStage = slashArgumentMode === 'mixed' || slashArgumentMode === 'text'

  const closeTrigger = () => {
    setTrigger(null)
    setTriggerItems([])
    resetTriggerActive()
  }

  /** Step the highlight, marking it as the user's own deliberate pick. */
  const moveTriggerActive = (delta: number) => {
    setTriggerActiveExplicit(true)
    setTriggerActive(idx => (idx + delta + triggerItems.length) % triggerItems.length)
  }

  useEffect(() => {
    setTriggerActive(idx => Math.min(idx, Math.max(0, triggerItems.length - 1)))
  }, [triggerItems.length])

  // Commit the literally-typed `/command arg` as a directive chip — used when
  // the completion list is empty because the arg is already fully typed (the
  // backend completer drops exact matches). Reuses the chip path via a
  // synthetic item whose serialized form is the verbatim text.
  const commitTypedSlashDirective = (): boolean => {
    if (trigger?.kind !== '/') {
      return false
    }

    // Free prose must stay ordinary contentEditable text. This guard also
    // protects against a stale completion result reaching the keydown path
    // before refreshTrigger has caught up with the latest DOM input.
    if (desktopSlashCommandArgumentMode(slashCommandToken(trigger.query)) !== 'options') {
      return false
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

    return true
  }

  const replaceTriggerWithChip = (item: Unstable_TriggerItem, options?: { descend?: boolean }) => {
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

    // Tab on a folder walks INTO it instead of committing it: re-type the
    // token as the bare path so the next `complete.path` lists that folder's
    // children, exactly as typing the path by hand would. Enter still commits
    // the folder itself — the two intents are distinct, so the keys are too.
    // Only `@` folders descend; a slash command's arg list has no hierarchy.
    const descendInto =
      options?.descend && trigger.kind === '@' && item.type === 'folder'
        ? String((item.metadata as { insertId?: unknown } | undefined)?.insertId ?? '')
        : ''

    if (descendInto) {
      const path = descendInto.endsWith('/') ? descendInto : `${descendInto}/`
      const current = composerPlainText(editor)
      const prefix = current.slice(0, Math.max(0, current.length - trigger.tokenLength))

      renderComposerContents(editor, `${prefix}@${path}`)
      placeCaretEnd(editor)
      draftRef.current = composerPlainText(editor)
      setComposerText(draftRef.current)
      requestMainFocus()
      window.setTimeout(refreshTrigger, 0)

      return
    }

    // Picking a bare arg-taking command (e.g. `/personality`) shouldn't commit
    // it — expand to its options step so the popover shows the inline list, just
    // as typing `/personality ` by hand would. A serialized value with a space is
    // already an arg pick (`/personality alice`), so it commits normally. An
    // inline (mid-message) pick never expands: it's a reference inside prose, so
    // there's no command invocation for the args to belong to.
    const command = (item.metadata as { command?: string } | undefined)?.command ?? ''

    const argumentMode = desktopSlashCommandArgumentMode(command)
    const expandsToArgs = trigger.kind === '/' && !trigger.inline && !serialized.includes(' ') && argumentMode !== null

    const text = starter || serialized.endsWith(' ') ? serialized : `${serialized} `
    const directive = !starter && serialized.match(/^@([^:]+):(.+)$/)
    // No pill while expanding — the bare command stays plain text until an arg
    // is picked, at which point a single pill is emitted for the full command.
    const slashKind = !expandsToArgs && trigger.kind === '/' ? slashChipKindForItem(item) : null
    const keepTriggerOpen = starter || (expandsToArgs && argumentMode !== 'text')

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

  /** Backspace inside an `@` path drops the last segment (`a/b/` → `a/`)
   *  instead of one character. Descending is one Tab per level, so climbing
   *  back out should cost one key too rather than a held delete. Returns
   *  false when the caret isn't in a path, so keydown falls through. */
  const ascendTriggerPath = () => {
    const editor = editorRef.current

    if (!editor || trigger?.kind !== '@' || !trigger.query.includes('/')) {
      return false
    }

    // Trailing slash means we're listing a folder's children: drop that
    // folder. Otherwise a partial segment is typed — drop just that.
    const trimmed = trigger.query.replace(/\/$/, '')
    const parent = trimmed.slice(0, trimmed.lastIndexOf('/') + 1)

    const current = composerPlainText(editor)
    const prefix = current.slice(0, Math.max(0, current.length - trigger.tokenLength))

    renderComposerContents(editor, `${prefix}@${parent}`)
    placeCaretEnd(editor)
    draftRef.current = composerPlainText(editor)
    setComposerText(draftRef.current)
    window.setTimeout(refreshTrigger, 0)

    return true
  }

  return {
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
  }
}
