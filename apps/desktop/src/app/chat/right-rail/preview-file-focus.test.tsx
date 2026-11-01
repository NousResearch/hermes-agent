import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { composerFocusKeysAllowed } from '@/lib/keybinds/composer-focus-keys'

import { MarkdownPreview, SourceView } from './preview-file'

vi.mock('@/components/chat/shiki-highlighter', () => ({
  LazyShiki: ({ code }: { code: string }) => <pre>{code}</pre>
}))

afterEach(() => {
  cleanup()
  window.getSelection()?.removeAllRanges()
})

function selectAll(node: HTMLElement, modifier: 'ctrlKey' | 'metaKey') {
  return fireEvent.keyDown(node, { key: 'a', [modifier]: true })
}

describe('file preview keyboard ownership', () => {
  it('focuses source locally and copies the entire virtualized file without gutters', () => {
    const text = Array.from({ length: 1600 }, (_, n) => `source line ${n + 1}`).join('\n')

    const { container } = render(
      <>
        <textarea aria-label="Composer" defaultValue="chat draft" />
        <SourceView filePath="/workspace/file.txt" language="text" text={text} />
      </>
    )

    const composer = screen.getByRole('textbox', { name: 'Composer' })
    composer.focus()
    const firstLine = container.querySelector('.preview-source-code pre') as HTMLElement
    const scroller = firstLine.closest('.overflow-auto') as HTMLElement
    expect(container.textContent).not.toContain('source line 1600')

    fireEvent.mouseDown(firstLine, { button: 0 })
    expect(window.document.activeElement).toBe(scroller)
    const typing = new KeyboardEvent('keydown', { bubbles: true, key: 'x' })
    firstLine.dispatchEvent(typing)
    expect(composerFocusKeysAllowed(typing, 'type')).toBe(false)

    for (const modifier of ['ctrlKey', 'metaKey'] as const) {
      expect(selectAll(scroller, modifier)).toBe(false)
      const setData = vi.fn()
      expect(fireEvent.copy(scroller, { clipboardData: { setData } })).toBe(false)
      expect(setData).toHaveBeenCalledWith('text/plain', text)
    }

    const partial = window.document.createRange()
    partial.selectNodeContents(firstLine)
    partial.setEnd(firstLine.firstChild!, 6)
    window.getSelection()?.removeAllRanges()
    window.getSelection()?.addRange(partial)
    const setData = vi.fn()
    expect(fireEvent.copy(scroller, { clipboardData: { setData } })).toBe(true)
    expect(setData).not.toHaveBeenCalled()
    const partialText = window.getSelection()?.toString()
    const claimed = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ctrlKey: true, key: 'a' })
    claimed.preventDefault()
    scroller.dispatchEvent(claimed)
    expect(window.getSelection()?.toString()).toBe(partialText)
    expect(fireEvent.keyDown(scroller, { ctrlKey: true, isComposing: true, key: 'a' })).toBe(true)
    expect(window.getSelection()?.toString()).toBe(partialText)

    composer.focus()
    expect(window.document.activeElement).toBe(composer)
    expect(selectAll(composer, 'metaKey')).toBe(true)
  })

  it('scopes rendered Markdown selection and never focuses on mount, reload, or tab replacement', async () => {
    const view = (text: string, tab = 'a') => (
      <>
        <textarea aria-label="Composer" defaultValue="chat draft" />
        <MarkdownPreview key={tab} text={text} />
      </>
    )

    const { rerender } = render(view('# First document\n\nFile body'))
    const composer = screen.getByRole('textbox', { name: 'Composer' })
    composer.focus()
    await act(async () => rerender(view('# Updated document\n\nFile body')))
    expect(window.document.activeElement).toBe(composer)

    const paragraph = screen.getByText('File body')
    fireEvent.mouseDown(paragraph, { button: 0 })
    const surface = paragraph.closest('[data-preview-markdown]') as HTMLElement
    expect(window.document.activeElement).toBe(surface)
    expect(selectAll(surface, 'metaKey')).toBe(false)
    expect(window.getSelection()?.toString()).toContain('File body')
    expect(window.getSelection()?.toString()).not.toContain('chat draft')

    // Links and controls retain their own native focus/activation behavior.
    rerender(view('[documentation](https://example.com)'))
    const link = screen.getByRole('link', { name: 'documentation' })
    link.focus()
    fireEvent.mouseDown(link, { button: 0 })
    expect(window.document.activeElement).toBe(link)
    expect(selectAll(link, 'metaKey')).toBe(false)
    expect(window.getSelection()?.toString()).toContain('documentation')

    composer.focus()
    rerender(view('# Another tab\n\nOther body', 'b'))
    expect(window.document.activeElement).toBe(composer)
    const other = screen.getByText('Other body')
    fireEvent.mouseDown(other, { button: 0 })
    expect(window.document.activeElement).toBe(other.closest('[data-preview-markdown]'))
  })
})
