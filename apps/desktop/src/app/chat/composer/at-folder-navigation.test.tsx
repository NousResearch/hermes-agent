import type { Unstable_TriggerItem } from '@assistant-ui/core'
import { act, renderHook } from '@testing-library/react'
import { createRef } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { useComposerTrigger } from './hooks/use-composer-trigger'
import { composerPlainText, renderComposerContents, RICH_INPUT_SLOT } from './rich-editor'

/**
 * Folder navigation in the `@` popover, driven through the REAL hook against a
 * real contentEditable.
 *
 * Tab and Enter used to be the same branch, so picking a folder always
 * committed a chip and closed the menu — there was no way to walk into a
 * subdirectory from the list. Tab now descends, Enter still commits, and
 * Backspace climbs back out one segment.
 */
function folderItem(path: string): Unstable_TriggerItem {
  return {
    id: `@folder:${path}|0`,
    type: 'folder',
    label: path.split('/').filter(Boolean).pop() ?? path,
    metadata: {
      icon: 'folder',
      display: path,
      meta: 'dir',
      rawText: `@folder:${path}`,
      insertId: path
    }
  }
}

function setup(initialText: string) {
  const editor = document.createElement('div')
  editor.contentEditable = 'true'
  // The real composer marks its editor with this slot; `composerPlainText`
  // keys off it to decide whether a DIV contributes a trailing newline.
  // Without it the harness would silently diverge from production text.
  editor.dataset.slot = RICH_INPUT_SLOT
  document.body.append(editor)
  renderComposerContents(editor, initialText)

  // Caret at the end, which is where a typed trigger always leaves it.
  const range = document.createRange()
  range.selectNodeContents(editor)
  range.collapse(false)
  const sel = window.getSelection()
  sel?.removeAllRanges()
  sel?.addRange(range)

  const editorRef = createRef<HTMLDivElement>() as { current: HTMLDivElement | null }
  editorRef.current = editor

  const draftRef = { current: initialText }
  const setComposerText = vi.fn()

  const { result } = renderHook(() =>
    useComposerTrigger({
      at: { adapter: null, loading: false },
      draftRef,
      editorRef,
      requestMainFocus: vi.fn(),
      setComposerText,
      slash: { adapter: null, loading: false }
    })
  )

  act(() => {
    result.current.refreshTrigger()
  })

  return { editor, result, setComposerText }
}

describe('@ folder navigation', () => {
  it('Tab on a folder walks into it and keeps the popover open', () => {
    const { editor, result } = setup('@app')

    expect(result.current.trigger).toMatchObject({ kind: '@', query: 'app' })

    act(() => {
      result.current.replaceTriggerWithChip(folderItem('apps'), { descend: true })
    })

    // Plain text, not a chip — the token is still being typed.
    expect(composerPlainText(editor)).toBe('@apps/')
    expect(editor.querySelector('[data-ref-text]')).toBeNull()
  })

  it('descends repeatedly, one level per Tab', () => {
    const { editor, result } = setup('@apps/desk')

    act(() => {
      result.current.replaceTriggerWithChip(folderItem('apps/desktop'), { descend: true })
    })

    expect(composerPlainText(editor)).toBe('@apps/desktop/')
  })

  it('Enter on a folder commits it as a chip instead of descending', () => {
    const { editor, result } = setup('@app')

    act(() => {
      result.current.replaceTriggerWithChip(folderItem('apps'))
    })

    const chip = editor.querySelector('[data-ref-text]')

    expect(chip).not.toBeNull()
    expect(chip?.getAttribute('data-ref-kind')).toBe('folder')
    expect(composerPlainText(editor)).toContain('@folder:')
  })

  it('only descends for `@` folders — a file pick still commits', () => {
    const { editor, result } = setup('@main')

    const file: Unstable_TriggerItem = {
      id: '@file:src/main.tsx|0',
      type: 'file',
      label: 'main.tsx',
      metadata: {
        icon: 'file',
        display: 'main.tsx',
        meta: 'src',
        rawText: '@file:src/main.tsx',
        insertId: 'src/main.tsx'
      }
    }

    act(() => {
      result.current.replaceTriggerWithChip(file, { descend: true })
    })

    expect(editor.querySelector('[data-ref-kind="file"]')).not.toBeNull()
  })

  it('Backspace climbs out one segment at a time', () => {
    const { editor, result } = setup('@apps/desktop/')

    let handled = false
    act(() => {
      handled = result.current.ascendTriggerPath()
    })

    expect(handled).toBe(true)
    expect(composerPlainText(editor)).toBe('@apps/')
  })

  it('Backspace drops a partially typed segment before its parent', () => {
    const { editor, result } = setup('@apps/desk')

    act(() => {
      result.current.ascendTriggerPath()
    })

    expect(composerPlainText(editor)).toBe('@apps/')
  })

  it('leaves Backspace alone when there is no path to climb', () => {
    const { result } = setup('@apps')

    let handled = true
    act(() => {
      handled = result.current.ascendTriggerPath()
    })

    // No `/` in the query — normal character deletion must still happen.
    expect(handled).toBe(false)
  })

  it('preserves text typed before the mention', () => {
    const { editor, result } = setup('look at @app')

    act(() => {
      result.current.replaceTriggerWithChip(folderItem('apps'), { descend: true })
    })

    expect(composerPlainText(editor)).toBe('look at @apps/')
  })
})
