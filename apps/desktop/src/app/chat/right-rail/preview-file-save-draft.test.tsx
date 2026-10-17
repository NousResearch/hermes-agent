import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'

import { EditorView } from '@codemirror/view'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { LocalFilePreview } from './preview-file'

vi.mock('@/components/chat/shiki-highlighter', () => ({
  LazyShiki: ({ code }: { code: string }) => <pre>{code}</pre>
}))

it('keeps edits made during a pending save and saves them against the committed baseline', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'hermes-preview-draft-'))
  const filePath = path.join(directory, 'note.txt')
  const rangeDescriptors = Object.getOwnPropertyDescriptors(Range.prototype)
  let releaseWrite!: () => void

  let pauseWrite: Promise<void> | undefined = new Promise(resolve => {
    releaseWrite = resolve
  })

  const writeTextFile = vi.fn(async (file: string, content: string) => {
    await writeFile(file, content)
    await pauseWrite

    return { path: file }
  })

  try {
    await writeFile(filePath, 'original')
    $connection.set(null)
    vi.stubGlobal('hermesDesktop', {
      readFileText: async (file: string) => ({
        binary: false,
        path: file,
        text: await readFile(file, 'utf8'),
        truncated: false
      }),
      writeTextFile
    })
    // CodeMirror is real; only selection geometry is absent from jsdom.
    Object.defineProperties(Range.prototype, {
      getClientRects: { configurable: true, value: () => [] },
      getBoundingClientRect: { configurable: true, value: () => new DOMRect() }
    })

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
    act(() => editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: 'first draft' } }))
    fireEvent.click(screen.getByRole('button', { name: /Save/ }))
    await waitFor(async () => expect(await readFile(filePath, 'utf8')).toBe('first draft'))

    // The native/remote write has the old bytes, while the editable buffer advances.
    act(() => editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: 'newer draft' } }))
    await act(async () => {
      pauseWrite = undefined
      releaseWrite()
    })

    expect(editor.dom.isConnected).toBe(true)
    expect(editor.state.doc.toString()).toBe('newer draft')
    expect(await readFile(filePath, 'utf8')).toBe('first draft')
    const save = screen.getByRole('button', { name: /Save/ }) as HTMLButtonElement
    expect(save.disabled).toBe(false)

    fireEvent.click(save)
    await screen.findByRole('button', { name: 'Edit' })
    expect(writeTextFile).toHaveBeenCalledTimes(2)
    expect(await readFile(filePath, 'utf8')).toBe('newer draft')
  } finally {
    releaseWrite()
    cleanup()
    vi.unstubAllGlobals()
    $connection.set(null)

    for (const key of ['getClientRects', 'getBoundingClientRect']) {
      if (rangeDescriptors[key]) {
        Object.defineProperty(Range.prototype, key, rangeDescriptors[key])
      } else {
        Reflect.deleteProperty(Range.prototype, key)
      }
    }

    await rm(directory, { recursive: true, force: true })
  }
})
