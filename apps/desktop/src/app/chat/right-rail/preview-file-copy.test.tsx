import { EditorView } from '@codemirror/view'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { LocalFilePreview } from './preview-file'

vi.mock('@/components/chat/shiki-highlighter', () => ({
  LazyShiki: ({ code }: { code: string }) => <pre>{code}</pre>
}))

const text = '<html>\r\n  Привет 👋\r\n</html>\r\n'

const target = {
  kind: 'file' as const,
  label: 'PASTE-IN-HTML.TXT',
  path: '/document.txt',
  previewKind: 'text' as const,
  source: '/document.txt',
  url: 'file:///document.txt'
}

function setup(truncated = false, textOverride: string = text) {
  const writeClipboard = vi.fn().mockResolvedValue(undefined)
  vi.stubGlobal('hermesDesktop', {
    readFileText: vi.fn().mockResolvedValue({ text: textOverride, binary: false, truncated }),
    writeClipboard
  })

  return { ...render(<LocalFilePreview reloadKey={0} target={target} />), writeClipboard }
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

it('copies the entire original source next to Edit, preserving whitespace and without line numbers', async () => {
  const { writeClipboard } = setup()
  await screen.findByRole('button', { name: 'Edit' })
  fireEvent.click(screen.getByRole('button', { name: 'Copy entire document' }))
  await waitFor(() => expect(writeClipboard).toHaveBeenCalledWith(text))
  await screen.findByRole('button', { name: 'Copied' })
})

it('reports clipboard rejection and lets the user retry', async () => {
  const { writeClipboard } = setup()
  writeClipboard.mockRejectedValueOnce(new Error('denied'))
  fireEvent.click(await screen.findByRole('button', { name: 'Copy entire document' }))
  expect((await screen.findByRole('alert')).textContent).toBe('Copy failed')
  expect(screen.queryByRole('button', { name: 'Copied' })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Copy entire document' }))
  await screen.findByRole('button', { name: 'Copied' })
  expect(screen.queryByRole('alert')).toBeNull()
})

it('copies the latest unsaved editor buffer, including changes after the dirty flag is set', async () => {
  const { container, writeClipboard } = setup()
  Object.defineProperties(Range.prototype, {
    getClientRects: { configurable: true, value: () => [] },
    getBoundingClientRect: { configurable: true, value: () => new DOMRect() }
  })
  fireEvent.click(await screen.findByRole('button', { name: 'Edit' }))
  const editor = EditorView.findFromDOM(container.querySelector('.cm-content')!)!
  act(() => editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: 'first draft' } }))
  act(() => editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: 'latest draft\n\t終\n' } }))
  fireEvent.click(screen.getByRole('button', { name: 'Copy entire document' }))
  await waitFor(() => expect(writeClipboard).toHaveBeenCalledWith('latest draft\n\t終\n'))
})

it('never presents a truncated preview as an entire document', async () => {
  setup(true)
  await screen.findByText('Showing first 512 KB.')
  expect(screen.queryByRole('button', { name: 'Copy entire document' })).toBeNull()
})

it('prevents overlapping clipboard writes and only confirms after resolution', async () => {
  const { writeClipboard } = setup()
  let resolve!: () => void
  writeClipboard.mockImplementationOnce(
    () =>
      new Promise<void>(done => {
        resolve = done
      })
  )
  const button = await screen.findByRole('button', { name: 'Copy entire document' })
  fireEvent.click(button)
  fireEvent.click(button)
  expect(writeClipboard).toHaveBeenCalledTimes(1)
  expect((button as HTMLButtonElement).disabled).toBe(true)
  expect(screen.queryByRole('button', { name: 'Copied' })).toBeNull()
  await act(async () => resolve())
  await screen.findByRole('button', { name: 'Copied' })
})

it('shows the copied text in a green confirmation block', async () => {
  setup()
  await screen.findByRole('button', { name: 'Edit' })
  fireEvent.click(screen.getByRole('button', { name: 'Copy entire document' }))
  const confirmation = await screen.findByRole('status')
  expect(confirmation.textContent).toContain(text)
  expect(confirmation.getAttribute('data-slot')).toBe('copy-document-confirmation')
})

it('caps the confirmation preview for a long document and says how much was omitted', async () => {
  const long = `${'a'.repeat(9000)}\n${'b'.repeat(9000)}`
  setup(false, long)
  await screen.findByRole('button', { name: 'Edit' })
  fireEvent.click(screen.getByRole('button', { name: 'Copy entire document' }))
  const confirmation = await screen.findByRole('status')
  const preview = confirmation.querySelector('pre')!.textContent ?? ''
  expect(preview.length).toBeLessThan(long.length)
  expect(confirmation.textContent).toContain(String(long.length))
})
