import { render } from '@testing-library/react'
import { createRef, type RefObject } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { useComposerUndo } from './use-composer-undo'

/** Mount the hook against a real contentEditable, exposing its API. */
function mountUndo(editorRef: RefObject<HTMLDivElement | null>, onSync: () => string) {
  const api: { current: ReturnType<typeof useComposerUndo> | null } = { current: null }

  const Harness = () => {
    // Assigned during render on purpose: the tests drive the API imperatively
    // right after mount, and this is a harness, not app state.
    api.current = useComposerUndo({ editorRef, syncDraftFromEditor: onSync })

    return null
  }

  const view = render(<Harness />)

  return { api, view }
}

function makeEditor(text: string) {
  const editor = document.createElement('div')
  editor.contentEditable = 'true'
  // jsdom only focuses a contentEditable div when it's explicitly focusable;
  // the real editor is reachable via the composer's focus bus.
  editor.tabIndex = 0
  editor.append(document.createTextNode(text))
  document.body.append(editor)

  const ref = createRef<HTMLDivElement>() as RefObject<HTMLDivElement | null>
  ref.current = editor

  return { editor, ref }
}

const caretAtEnd = (editor: HTMLElement) => {
  const range = document.createRange()
  const selection = window.getSelection()!
  range.selectNodeContents(editor)
  range.collapse(false)
  selection.removeAllRanges()
  selection.addRange(range)
}

describe('useComposerUndo', () => {
  it('restores the pre-edit text, which is what a paste destroyed', () => {
    const { editor, ref } = makeEditor('before')
    caretAtEnd(editor)

    const { api, view } = mountUndo(ref, () => editor.textContent || '')

    // Bank, then simulate the Range-based paste that Chromium never records.
    api.current!.recordUndoPoint()
    editor.append(document.createTextNode(' PASTED'))
    expect(editor.textContent).toBe('before PASTED')

    api.current!.undo()
    expect(editor.textContent).toBe('before')

    api.current!.redo()
    expect(editor.textContent).toBe('before PASTED')

    view.unmount()
    editor.remove()
  })

  it('withUndoPoint banks only when the edit actually ran', () => {
    const { editor, ref } = makeEditor('text')
    caretAtEnd(editor)

    const { api, view } = mountUndo(ref, () => editor.textContent || '')

    // A guard that declines must not consume an undo slot.
    expect(api.current!.withUndoPoint(() => false)).toBe(false)
    expect(api.current!.undo()).toBe(false)

    expect(
      api.current!.withUndoPoint(() => {
        editor.append(document.createTextNode('!'))

        return true
      })
    ).toBe(true)

    api.current!.undo()
    expect(editor.textContent).toBe('text')

    view.unmount()
    editor.remove()
  })

  it('claims a native historyUndo aimed at the focused editor', () => {
    const { editor, ref } = makeEditor('kept')
    editor.focus()
    caretAtEnd(editor)

    const { api, view } = mountUndo(ref, () => editor.textContent || '')

    api.current!.recordUndoPoint()
    editor.append(document.createTextNode(' extra'))

    // What Electron's Edit menu `{ role: 'undo' }` produces.
    const event = new InputEvent('beforeinput', { bubbles: true, cancelable: true, inputType: 'historyUndo' })
    editor.dispatchEvent(event)

    expect(event.defaultPrevented).toBe(true)
    expect(editor.textContent).toBe('kept')

    view.unmount()
    editor.remove()
  })

  it('ignores a historyUndo while another editor holds focus', () => {
    const { editor, ref } = makeEditor('mine')
    const { editor: other } = makeEditor('theirs')

    other.focus()

    const { api, view } = mountUndo(ref, () => editor.textContent || '')

    api.current!.recordUndoPoint()
    editor.append(document.createTextNode(' changed'))

    const event = new InputEvent('beforeinput', { bubbles: true, cancelable: true, inputType: 'historyUndo' })
    other.dispatchEvent(event)

    // Not ours to claim — the other surface keeps its native behavior.
    expect(event.defaultPrevented).toBe(false)
    expect(editor.textContent).toBe('mine changed')

    view.unmount()
    editor.remove()
    other.remove()
  })

  it('keeps two mounted composers independent', () => {
    const { editor: main, ref: mainRef } = makeEditor('main')
    const { editor: edit, ref: editRef } = makeEditor('edit')

    const mainUndo = mountUndo(mainRef, () => main.textContent || '')
    const editUndo = mountUndo(editRef, () => edit.textContent || '')

    mainUndo.api.current!.recordUndoPoint()
    main.append(document.createTextNode(' typed'))

    // Undoing in the edit composer must not touch the main composer's text.
    editUndo.api.current!.undo()
    expect(main.textContent).toBe('main typed')

    mainUndo.api.current!.undo()
    expect(main.textContent).toBe('main')
    expect(edit.textContent).toBe('edit')

    mainUndo.view.unmount()
    editUndo.view.unmount()
    main.remove()
    edit.remove()
  })

  it('reset drops history so undo cannot cross a draft swap', () => {
    const { editor, ref } = makeEditor('session A')
    caretAtEnd(editor)

    const { api, view } = mountUndo(ref, () => editor.textContent || '')

    api.current!.recordUndoPoint()
    editor.append(document.createTextNode(' edited'))
    api.current!.resetUndoHistory()

    expect(api.current!.undo()).toBe(false)
    expect(editor.textContent).toBe('session A edited')

    view.unmount()
    editor.remove()
  })

  it('is inert when the editor ref is empty', () => {
    const ref = createRef<HTMLDivElement>() as RefObject<HTMLDivElement | null>
    const sync = vi.fn(() => '')

    const { api, view } = mountUndo(ref, sync)

    api.current!.recordUndoPoint()

    expect(api.current!.undo()).toBe(false)
    expect(sync).not.toHaveBeenCalled()

    view.unmount()
  })
})
