import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'

import { EditorView } from '@codemirror/view'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { LocalFilePreview } from './preview-file'

const rangeDescriptors = Object.getOwnPropertyDescriptors(Range.prototype)

// Syntax coloring is unrelated to saving; keep the real editor and filesystem.
vi.mock('@/components/chat/shiki-highlighter', () => ({
  LazyShiki: ({ code }: { code: string }) => <pre>{code}</pre>
}))

describe('preview editor save validation', () => {
  let directory: string
  let filePath: string
  let readError: Error | undefined

  const writeTextFile = vi.fn(async (file: string, content: string) => {
    await writeFile(file, content)

    return { path: file }
  })

  beforeEach(async () => {
    directory = await mkdtemp(path.join(tmpdir(), 'hermes-preview-save-'))
    filePath = path.join(directory, 'note.txt')
    await writeFile(filePath, 'original')
    readError = undefined
    writeTextFile.mockClear()
    $connection.set(null)
    vi.stubGlobal('hermesDesktop', {
      readFileText: async (file: string) => {
        if (readError) {
          throw readError
        }

        const bytes = await readFile(file)

        return {
          binary: bytes.includes(0),
          byteSize: bytes.length,
          path: file,
          text: bytes.toString('utf8'),
          truncated: false
        }
      },
      writeTextFile
    })
    // CodeMirror measures selection geometry; jsdom does not implement it.
    Object.defineProperties(Range.prototype, {
      getClientRects: { configurable: true, value: () => [] },
      getBoundingClientRect: { configurable: true, value: () => new DOMRect() }
    })
  })

  afterEach(async () => {
    cleanup()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()

    for (const key of ['getClientRects', 'getBoundingClientRect']) {
      if (rangeDescriptors[key]) {
        Object.defineProperty(Range.prototype, key, rangeDescriptors[key])
      } else {
        Reflect.deleteProperty(Range.prototype, key)
      }
    }

    $connection.set(null)
    await rm(directory, { recursive: true, force: true })
  })

  async function edit() {
    const rendered = render(
      <LocalFilePreview
        reloadKey={0}
        target={{
          kind: 'file',
          label: 'note.txt',
          path: filePath,
          previewKind: 'text',
          source: filePath,
          url: `file://${filePath}`
        }}
      />
    )

    fireEvent.click(await screen.findByRole('button', { name: 'Edit' }))
    const editor = EditorView.findFromDOM(rendered.container.querySelector('.cm-content')!)!
    act(() => editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: 'my draft' } }))

    return editor
  }

  it('keeps the draft and disk bytes when the validation read fails, then allows retry', async () => {
    const editor = await edit()
    await writeFile(filePath, 'external change')
    readError = new Error('validation read unavailable')
    fireEvent.click(screen.getByRole('button', { name: /Save/ }))

    await waitFor(() =>
      expect(
        screen.queryByText(/validation read unavailable/) || screen.queryByRole('button', { name: 'Edit' })
      ).not.toBeNull()
    )
    expect(screen.queryByText(/validation read unavailable/)).not.toBeNull()
    expect(writeTextFile).not.toHaveBeenCalled()
    expect(await readFile(filePath, 'utf8')).toBe('external change')
    expect(editor.state.doc.toString()).toBe('my draft')

    readError = undefined
    await writeFile(filePath, 'original')
    fireEvent.click(screen.getByRole('button', { name: /Save/ }))
    await waitFor(() => expect(writeTextFile).toHaveBeenCalledOnce())
    await screen.findByRole('button', { name: 'Edit' })
    expect(await readFile(filePath, 'utf8')).toBe('my draft')
  })

  it('requires explicit overwrite when the file becomes binary after editing begins', async () => {
    await edit()
    const replacement = Buffer.from([0, 1, 2, 3])
    await writeFile(filePath, replacement)
    fireEvent.click(screen.getByRole('button', { name: /Save/ }))

    await waitFor(() =>
      expect(
        screen.queryByRole('button', { name: /Overwrite/ }) || screen.queryByRole('button', { name: 'Edit' })
      ).not.toBeNull()
    )
    const overwrite = screen.getByRole('button', { name: /Overwrite/ })
    expect(writeTextFile).not.toHaveBeenCalled()
    expect(await readFile(filePath)).toEqual(replacement)

    fireEvent.click(overwrite)
    await screen.findByRole('button', { name: 'Edit' })
    expect(await readFile(filePath, 'utf8')).toBe('my draft')
  })
})
