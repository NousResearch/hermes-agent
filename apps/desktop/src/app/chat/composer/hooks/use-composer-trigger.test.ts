import type { Unstable_TriggerAdapter, Unstable_TriggerItem } from '@assistant-ui/core'
import { act, renderHook } from '@testing-library/react'
import { createRef } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { composerPlainText, renderComposerContents, RICH_INPUT_SLOT } from '../rich-editor'

import { useComposerTrigger } from './use-composer-trigger'

/** A live contentEditable seeded with `text`, caret parked at the end. */
function mountEditor(text: string) {
  const editor = document.createElement('div')
  editor.dataset.slot = RICH_INPUT_SLOT
  editor.contentEditable = 'true'
  document.body.append(editor)
  renderComposerContents(editor, text)

  const range = document.createRange()
  range.selectNodeContents(editor)
  range.collapse(false)
  const selection = window.getSelection()!
  selection.removeAllRanges()
  selection.addRange(range)

  return editor
}

const item = (command: string, group = 'Skills'): Unstable_TriggerItem => ({
  id: command,
  type: 'slash',
  label: command.slice(1),
  metadata: { command, display: command, meta: '', group, action: '', rawText: command }
})

function mountTrigger(editor: HTMLDivElement, items: Unstable_TriggerItem[]) {
  const editorRef = createRef<HTMLDivElement>() as { current: HTMLDivElement | null }
  editorRef.current = editor

  const draftRef = { current: composerPlainText(editor) }

  const adapter: Unstable_TriggerAdapter = {
    categories: () => [],
    categoryItems: () => [],
    search: () => items
  }

  const setComposerText = vi.fn()

  const hook = renderHook(() =>
    useComposerTrigger({
      at: { adapter: null, loading: false },
      draftRef,
      editorRef,
      requestMainFocus: vi.fn(),
      setComposerText,
      slash: { adapter, loading: false }
    })
  )

  return { draftRef, hook, setComposerText }
}

describe('useComposerTrigger — slash anywhere in the prompt', () => {
  it('opens the completion list for a slash typed mid-message', () => {
    const editor = mountEditor('please run /cle')
    const { hook } = mountTrigger(editor, [item('/clean')])

    act(() => hook.result.current.refreshTrigger())

    expect(hook.result.current.trigger).toMatchObject({ kind: '/', inline: true, query: 'cle' })
    expect(hook.result.current.triggerItems).toHaveLength(1)
  })

  it('inserts the picked skill inline and keeps the surrounding prose intact', () => {
    const editor = mountEditor('please run /cle')
    const { hook } = mountTrigger(editor, [item('/clean')])

    act(() => hook.result.current.refreshTrigger())
    act(() => hook.result.current.replaceTriggerWithChip(item('/clean')))

    // The `/cle` the user typed is replaced by the full command; "please run"
    // in front of it survives untouched.
    expect(composerPlainText(editor)).toBe('please run /clean ')
  })

  it('offers only skills mid-message, not app commands', () => {
    // `/model` and `/new` act on the app — meaningless as a reference in prose.
    const editor = mountEditor('please run /')
    const { hook } = mountTrigger(editor, [item('/clean'), item('/model', 'Commands'), item('/new', 'Commands')])

    act(() => hook.result.current.refreshTrigger())

    expect(hook.result.current.triggerItems.map(i => i.label)).toEqual(['clean'])
  })

  it('still offers the full command set at the start of the prompt', () => {
    const editor = mountEditor('/')
    const { hook } = mountTrigger(editor, [item('/clean'), item('/model', 'Commands')])

    act(() => hook.result.current.refreshTrigger())

    expect(hook.result.current.triggerItems.map(i => i.label)).toEqual(['clean', 'model'])
  })

  it('still opens the list for a slash at the start of the prompt', () => {
    const editor = mountEditor('/cle')
    const { hook } = mountTrigger(editor, [item('/clean')])

    act(() => hook.result.current.refreshTrigger())

    expect(hook.result.current.trigger).toMatchObject({ kind: '/', query: 'cle' })
    expect(hook.result.current.trigger?.inline).toBeUndefined()
  })

  it('leaves a mid-message file path alone', () => {
    const editor = mountEditor('open src/foo/bar')
    const { hook } = mountTrigger(editor, [item('/clean')])

    act(() => hook.result.current.refreshTrigger())

    expect(hook.result.current.trigger).toBeNull()
  })
})
