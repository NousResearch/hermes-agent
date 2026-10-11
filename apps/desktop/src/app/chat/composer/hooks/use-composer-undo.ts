import { type RefObject, useCallback, useEffect, useMemo, useRef } from 'react'

import { caretOffsetInEditor, composerPlainText, placeCaretAtOffset, renderComposerContents } from '../rich-editor'
import { type ComposerSnapshot, createComposerUndoHistory } from '../undo-history'

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

  // One physical ⌘Z can arrive as both keydown (composer handler) and
  // `beforeinput historyUndo` (Edit menu / main-process IPC). Apply a single
  // stack step for that turn so burst undo cannot collapse twice.
  const lastHistoryActionAt = useRef({ redo: 0, undo: 0 })

  const runHistoryAction = useCallback(
    (action: 'redo' | 'undo') => {
      const now = performance.now()

      if (now - lastHistoryActionAt.current[action] < 32) {
        return false
      }

      const ran = action === 'undo' ? undo() : redo()

      if (ran) {
        lastHistoryActionAt.current[action] = now
      }

      return ran
    },
    [redo, undo]
  )

  // Electron's Edit menu used to ship `{ role: 'undo' }`, whose accelerator the
  // macOS menu bar consumes before the web contents sees the keystroke (the
  // same hazard main.ts documents for ⌘W). Main now forwards ⌘Z via IPC and
  // the menu click shares that channel; this capture still claims a native
  // `historyUndo` aimed at the focused composer so menu and keystroke agree.
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
        runHistoryAction('undo')
      } else {
        runHistoryAction('redo')
      }
    }

    document.addEventListener('beforeinput', onBeforeInput, true)

    return () => document.removeEventListener('beforeinput', onBeforeInput, true)
  }, [editorRef, runHistoryAction])

  return {
    recordUndoPoint,
    redo: () => runHistoryAction('redo'),
    resetUndoHistory,
    undo: () => runHistoryAction('undo'),
    withUndoPoint
  }
}
