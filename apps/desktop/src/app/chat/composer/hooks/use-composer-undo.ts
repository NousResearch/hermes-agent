import { type RefObject, useCallback, useEffect, useMemo } from 'react'

import { caretOffsetInEditor, composerPlainText, placeCaretAtOffset, renderComposerContents } from '../rich-editor'
import { type ComposerSnapshot, createComposerUndoHistory } from '../undo-history'

/** Each mounted composer's undo recorder, keyed by its editor element, for the
 *  one edit the app makes from outside the composer tree: a context-menu
 *  replacement (`app/context-menu/composer-menu.ts`). */
const recorders = new WeakMap<HTMLElement, () => void>()

/** Bank `editor`'s current state on its composer's undo stack, if one is mounted. */
export function recordComposerUndoPoint(editor: HTMLElement): void {
  recorders.get(editor)?.()
}

interface UseComposerUndoArgs {
  editorRef: RefObject<HTMLDivElement | null>
  /** Push a restored snapshot back into draftRef + composer state. */
  syncDraftFromEditor: () => string
}

/**
 * Undo/redo for the rich composer.
 *
 * The editor mutates its DOM through `Range` to dodge Chromium's O(n²) editing
 * pipeline (#45812), which also dodges Chromium's undo stack — so a paste was
 * invisible to ⌘Z and the keystroke undid whatever edit came before it instead.
 * We own the stack outright rather than half of it: every edit path records the
 * pre-edit state here, and the editor claims ⌘Z / ⌘⇧Z itself.
 */
export function useComposerUndo({ editorRef, syncDraftFromEditor }: UseComposerUndoArgs) {
  const history = useMemo(() => createComposerUndoHistory(), [])

  const snapshot = useCallback((): ComposerSnapshot => {
    const editor = editorRef.current

    if (!editor) {
      return { caret: 0, text: '' }
    }

    return { caret: caretOffsetInEditor(editor), text: composerPlainText(editor) }
  }, [editorRef])

  /** Bank the current state before mutating the editor. `coalesce` marks a
   *  keystroke, so a run of typing collapses into one undo step. */
  const recordUndoPoint = useCallback(
    (options?: { coalesce?: boolean }) => {
      if (editorRef.current) {
        history.record(snapshot(), options)
      }
    },
    [editorRef, history, snapshot]
  )

  useEffect(() => {
    const editor = editorRef.current

    if (!editor) {
      return
    }

    const record = () => recordUndoPoint()

    recorders.set(editor, record)

    return () => {
      if (recorders.get(editor) === record) {
        recorders.delete(editor)
      }
    }
  }, [editorRef, recordUndoPoint])

  const applySnapshot = useCallback(
    (next: ComposerSnapshot | null) => {
      const editor = editorRef.current

      if (!next || !editor) {
        return false
      }

      renderComposerContents(editor, next.text)
      placeCaretAtOffset(editor, next.caret)
      syncDraftFromEditor()

      return true
    },
    [editorRef, syncDraftFromEditor]
  )

  /** Run a conditional edit, banking its pre-edit state only if it actually
   *  ran. The snapshot has to be taken first (the edit destroys the state we'd
   *  be saving), but recording unconditionally would clear the redo stack on
   *  every Backspace that falls through to the native path. */
  const withUndoPoint = useCallback(
    (edit: () => boolean) => {
      const before = snapshot()
      const ran = edit()

      if (ran) {
        history.record(before)
      }

      return ran
    },
    [history, snapshot]
  )

  const undo = useCallback(() => applySnapshot(history.undo(snapshot())), [applySnapshot, history, snapshot])
  const redo = useCallback(() => applySnapshot(history.redo(snapshot())), [applySnapshot, history, snapshot])

  // A session/draft swap makes prior history meaningless — undoing into another
  // conversation's text is worse than having no history at all.
  const resetUndoHistory = useCallback(() => history.reset(), [history])

  // Electron's Edit menu ships `{ role: 'undo' }`, whose accelerator the macOS
  // menu bar consumes before the web contents sees the keystroke (the same
  // hazard main.ts documents for ⌘W). It fires the native editing command,
  // which knows nothing about our stack. Claim it at the document level while
  // the composer holds focus, so the menu item and the keystroke agree.
  useEffect(() => {
    const onBeforeInput = (event: Event) => {
      const inputType = (event as InputEvent).inputType

      if (inputType !== 'historyUndo' && inputType !== 'historyRedo') {
        return
      }

      if (document.activeElement !== editorRef.current) {
        return
      }

      event.preventDefault()

      if (inputType === 'historyUndo') {
        undo()
      } else {
        redo()
      }
    }

    document.addEventListener('beforeinput', onBeforeInput, true)

    return () => document.removeEventListener('beforeinput', onBeforeInput, true)
  }, [editorRef, redo, undo])

  return { recordUndoPoint, redo, resetUndoHistory, undo, withUndoPoint }
}
