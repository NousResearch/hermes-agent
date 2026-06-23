import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useState } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

vi.mock('react-shiki', () => ({
  default: ({ children }: { children: string }) => <pre className="shiki">{children}</pre>
}))

import { computeTokenDiff, SourceView } from './preview-file'

describe('computeTokenDiff', () => {
  it('keeps shared tokens unaccented and emphasizes changed values', () => {
    const result = computeTokenDiff('const timeout = 1000', 'const timeout = 2500')

    expect(result.previous).toEqual([
      { changed: false, text: 'const timeout = ' },
      { changed: true, text: '1000' }
    ])
    expect(result.next).toEqual([
      { changed: false, text: 'const timeout = ' },
      { changed: true, text: '2500' }
    ])
  })

  it('handles punctuation-only edits without marking the whole line', () => {
    const result = computeTokenDiff('run(task)', 'run(task);')

    expect(result.next).toEqual([
      { changed: false, text: 'run(task)' },
      { changed: true, text: ';' }
    ])
  })
})

describe('SourceView save transition', () => {
  it('keeps the editor visible until updated highlighting is ready', async () => {
    let resolveWrite: ((result: { success: boolean }) => void) | undefined

    const writeFileText = vi.fn(
      () =>
        new Promise<{ success: boolean }>(resolve => {
          resolveWrite = resolve
        })
    )

    ;(window as unknown as { hermesDesktop: { writeFileText: typeof writeFileText } }).hermesDesktop = {
      writeFileText
    }

    function Harness() {
      const [text, setText] = useState('const value = 1')

      return (
        <I18nProvider configClient={null}>
          <SourceView filePath="/work/file.ts" language="typescript" onContentSaved={setText} text={text} />
        </I18nProvider>
      )
    }

    render(<Harness />)
    fireEvent.click(screen.getByRole('button', { name: 'Edit' }))

    const editor = screen.getByRole('textbox')

    fireEvent.change(editor, { target: { value: 'const value = 2' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    expect(editor.classList.contains('opacity-100')).toBe(true)
    expect((screen.getByRole('button', { name: 'Saving…' }) as HTMLButtonElement).disabled).toBe(true)

    await act(async () => {
      resolveWrite?.({ success: true })
    })

    expect(editor.classList.contains('opacity-100')).toBe(true)

    await waitFor(() => expect((screen.getByRole('button', { name: 'Edit' }) as HTMLButtonElement).disabled).toBe(false))
    expect(editor.classList.contains('opacity-0')).toBe(true)
    expect(screen.getAllByText('const value = 2')).toHaveLength(2)
  })
})
