import { EditorView } from '@codemirror/view'
import { cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { createRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { CodeEditorApi } from './code-editor'

const originalCreateRange = document.createRange.bind(document)

function suppressCodeMirrorGeometryError(event: ErrorEvent) {
  if (String(event.error || event.message).includes('getClientRects is not a function')) {
    event.preventDefault()
  }
}

async function renderEditor({
  initialValue = 'Alpha alpha Alpha',
  onCancel = vi.fn(),
  onChange = vi.fn()
}: {
  initialValue?: string
  onCancel?: () => void
  onChange?: (value: string) => void
} = {}) {
  const { CodeEditor } = await import('./code-editor')
  const apiRef = createRef<CodeEditorApi | null>()

  const rendered = render(
    <CodeEditor
      apiRef={apiRef}
      filePath="example.ts"
      initialValue={initialValue}
      onCancel={onCancel}
      onChange={onChange}
    />
  )

  const content = rendered.container.querySelector('.cm-content')

  expect(content).toBeInstanceOf(HTMLElement)

  return { ...rendered, apiRef, content: content as HTMLElement, onCancel, onChange }
}

function searchControls(container: HTMLElement) {
  const panel = container.querySelector('.cm-search')

  expect(panel).toBeInstanceOf(HTMLElement)

  return {
    caseSensitive: panel!.querySelector<HTMLInputElement>('input[name="case"]')!,
    close: panel!.querySelector<HTMLButtonElement>('button[name="close"]')!,
    next: panel!.querySelector<HTMLButtonElement>('button[name="next"]')!,
    previous: panel!.querySelector<HTMLButtonElement>('button[name="prev"]')!,
    regexp: panel!.querySelector<HTMLInputElement>('input[name="re"]')!,
    replace: panel!.querySelector<HTMLInputElement>('input[name="replace"]')!,
    replaceAll: panel!.querySelector<HTMLButtonElement>('button[name="replaceAll"]')!,
    replaceNext: panel!.querySelector<HTMLButtonElement>('button[name="replace"]')!,
    search: panel!.querySelector<HTMLInputElement>('input[name="search"]')!,
    selectAll: panel!.querySelector<HTMLButtonElement>('button[name="select"]')!,
    status: panel!.querySelector<HTMLElement>('.cm-search-match-status')!,
    wholeWord: panel!.querySelector<HTMLInputElement>('input[name="word"]')!
  }
}

describe('CodeEditor Find and Replace', () => {
  beforeEach(() => {
    const rect = new DOMRect(0, 0, 0, 0)

    const rectList = {
      0: rect,
      item: (index: number) => (index === 0 ? rect : null),
      length: 1,
      [Symbol.iterator]: function* () {
        yield rect
      }
    } as DOMRectList

    Object.defineProperty(document, 'createRange', {
      configurable: true,
      value: vi.fn(
        () =>
          ({
            collapse: vi.fn(),
            commonAncestorContainer: document.body,
            getBoundingClientRect: vi.fn(() => rect),
            getClientRects: vi.fn(() => rectList),
            setEnd: vi.fn(),
            setStart: vi.fn()
          }) as unknown as Range
      )
    })
    vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
      if (String(args[0]).includes('getClientRects is not a function')) {
        return
      }

      throw new Error(args.map(String).join(' '))
    })
    window.addEventListener('error', suppressCodeMirrorGeometryError)
  })

  afterEach(() => {
    cleanup()
    window.removeEventListener('error', suppressCodeMirrorGeometryError)
    vi.restoreAllMocks()
    Object.defineProperty(document, 'createRange', { configurable: true, value: originalCreateRange })
  })

  it('opens a complete native panel with Ctrl/Cmd+F', async () => {
    const rendered = await renderEditor()

    fireEvent.keyDown(rendered.content, { ctrlKey: true, key: 'f' })

    const controls = searchControls(rendered.container)

    expect(controls.search.getAttribute('aria-label')).toBe('Find')
    expect(controls.replace.getAttribute('aria-label')).toBe('Replace')
    expect(controls.next.textContent).toBe('Next')
    expect(controls.previous.textContent).toBe('Previous')
    expect(controls.selectAll.textContent).toBe('Select all')
    expect(controls.replaceNext.textContent).toBe('Replace')
    expect(controls.replaceAll.textContent).toBe('Replace all')
    expect(controls.caseSensitive).toBeInstanceOf(HTMLInputElement)
    expect(controls.wholeWord).toBeInstanceOf(HTMLInputElement)
    expect(controls.regexp).toBeInstanceOf(HTMLInputElement)
  })

  it('localizes native search and replacement announcements', async () => {
    const rendered = await renderEditor()
    const view = EditorView.findFromDOM(rendered.content)

    expect(view?.state.phrase('current match')).toBe('Current match')
    expect(view?.state.phrase('on line')).toBe('on line')
    expect(view?.state.phrase('replaced match on line $', 2)).toBe('Replaced match on line 2')
    expect(view?.state.phrase('replaced $ matches', 3)).toBe('Replaced 3 matches')
  })

  it('selects every match instead of collapsing Select all to one range', async () => {
    const rendered = await renderEditor({ initialValue: 'cat cat cat' })

    fireEvent.keyDown(rendered.content, { ctrlKey: true, key: 'f' })
    const controls = searchControls(rendered.container)
    fireEvent.change(controls.search, { target: { value: 'cat' } })
    await waitFor(() => expect(controls.status.textContent).toBe('1 of 3'))

    fireEvent.click(controls.selectAll)

    const view = EditorView.findFromDOM(rendered.content)

    expect(view?.state.selection.ranges).toHaveLength(3)
  })

  it('opens Find and Replace with Ctrl+H and focuses the replacement field', async () => {
    const rendered = await renderEditor()

    fireEvent.keyDown(rendered.content, { ctrlKey: true, key: 'h' })

    await waitFor(() => {
      const controls = searchControls(rendered.container)
      expect(document.activeElement).toBe(controls.replace)
    })
  })

  it('exposes the same Find and Replace panel through the editor toolbar API', async () => {
    const rendered = await renderEditor()

    expect(rendered.apiRef.current?.findReplace()).toBe(true)

    await waitFor(() => {
      const controls = searchControls(rendered.container)
      expect(document.activeElement).toBe(controls.replace)
    })
  })

  it('reports current and total matches while navigating with next and previous', async () => {
    const rendered = await renderEditor({ initialValue: 'cat cat cat' })

    fireEvent.keyDown(rendered.content, { ctrlKey: true, key: 'f' })
    const controls = searchControls(rendered.container)
    fireEvent.change(controls.search, { target: { value: 'cat' } })

    await waitFor(() => expect(controls.status.textContent).toBe('1 of 3'))

    fireEvent.click(controls.next)
    await waitFor(() => expect(controls.status.textContent).toBe('2 of 3'))

    fireEvent.click(controls.previous)
    await waitFor(() => expect(controls.status.textContent).toBe('1 of 3'))
  })

  it('navigates with Enter, Shift+Enter, F3, and Shift+F3 and wraps around', async () => {
    const rendered = await renderEditor({ initialValue: 'cat cat cat' })

    fireEvent.keyDown(rendered.content, { ctrlKey: true, key: 'f' })
    const controls = searchControls(rendered.container)
    fireEvent.change(controls.search, { target: { value: 'cat' } })
    await waitFor(() => expect(controls.status.textContent).toBe('1 of 3'))

    fireEvent.keyDown(controls.search, { key: 'Enter', keyCode: 13 })
    await waitFor(() => expect(controls.status.textContent).toBe('2 of 3'))

    fireEvent.keyDown(controls.search, { key: 'Enter', keyCode: 13, shiftKey: true })
    await waitFor(() => expect(controls.status.textContent).toBe('1 of 3'))

    fireEvent.keyDown(rendered.content, { key: 'F3', shiftKey: true })
    await waitFor(() => expect(controls.status.textContent).toBe('3 of 3'))

    fireEvent.keyDown(rendered.content, { key: 'F3' })
    await waitFor(() => expect(controls.status.textContent).toBe('1 of 3'))
  })

  it('supports whole-word, match-case, and regular-expression searches', async () => {
    const rendered = await renderEditor({ initialValue: 'cat scatter CAT cot cut dog' })

    fireEvent.keyDown(rendered.content, { ctrlKey: true, key: 'f' })
    const controls = searchControls(rendered.container)
    fireEvent.change(controls.search, { target: { value: 'cat' } })
    fireEvent.click(controls.wholeWord)
    await waitFor(() => expect(controls.status.textContent).toBe('1 of 2'))

    fireEvent.click(controls.caseSensitive)
    await waitFor(() => expect(controls.status.textContent).toBe('1 of 1'))

    fireEvent.click(controls.caseSensitive)
    fireEvent.change(controls.search, { target: { value: 'c.t' } })
    fireEvent.click(controls.regexp)
    await waitFor(() => expect(controls.status.textContent).toBe('1 of 4'))
  })

  it('replaces the current match and advances to the next result', async () => {
    const onChange = vi.fn()
    const rendered = await renderEditor({ initialValue: 'one two one', onChange })

    fireEvent.keyDown(rendered.content, { ctrlKey: true, key: 'f' })
    const controls = searchControls(rendered.container)
    fireEvent.change(controls.search, { target: { value: 'one' } })
    fireEvent.change(controls.replace, { target: { value: 'ONE' } })
    fireEvent.click(controls.caseSensitive)
    await waitFor(() => expect(rendered.container.querySelector('.cm-searchMatch-selected')?.textContent).toBe('one'))
    fireEvent.click(controls.replaceNext)

    await waitFor(() => expect(onChange).toHaveBeenLastCalledWith('ONE two one'))
    await waitFor(() => expect(controls.status.textContent).toBe('1 of 1'))
  })

  it('shows no-results and invalid-regex feedback without crashing', async () => {
    const rendered = await renderEditor({ initialValue: 'cat cat cat' })

    fireEvent.keyDown(rendered.content, { ctrlKey: true, key: 'f' })
    const controls = searchControls(rendered.container)
    fireEvent.change(controls.search, { target: { value: 'dog' } })
    await waitFor(() => expect(controls.status.textContent).toBe('No results'))

    fireEvent.change(controls.search, { target: { value: '(' } })
    fireEvent.click(controls.regexp)
    await waitFor(() => expect(controls.status.textContent).toBe('Invalid regular expression'))
  })

  it('replaces matching case-sensitive occurrences and undoes Replace All in one step', async () => {
    const onChange = vi.fn()
    const rendered = await renderEditor({ onChange })

    fireEvent.keyDown(rendered.content, { ctrlKey: true, key: 'f' })
    const controls = searchControls(rendered.container)
    fireEvent.change(controls.search, { target: { value: 'Alpha' } })
    fireEvent.change(controls.replace, { target: { value: 'Beta' } })
    fireEvent.click(controls.caseSensitive)
    fireEvent.click(controls.replaceAll)

    await waitFor(() => expect(onChange).toHaveBeenLastCalledWith('Beta alpha Beta'))

    fireEvent.keyDown(rendered.content, { ctrlKey: true, key: 'z' })
    await waitFor(() => expect(onChange).toHaveBeenLastCalledWith('Alpha alpha Alpha'))
  })

  it('closes search with Escape before invoking the file editor cancel action', async () => {
    const onCancel = vi.fn()
    const rendered = await renderEditor({ onCancel })

    fireEvent.keyDown(rendered.content, { ctrlKey: true, key: 'f' })
    const controls = searchControls(rendered.container)
    fireEvent.keyDown(controls.search, { key: 'Escape' })

    await waitFor(() => expect(rendered.container.querySelector('.cm-search')).toBeNull())
    expect(onCancel).not.toHaveBeenCalled()
  })
})
