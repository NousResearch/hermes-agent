import { cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import type { ClipboardEvent } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { insertPlainTextAtCaret } from './rich-editor'
import { shouldTryHostClipboardImage } from './text-utils'

// No global setupFiles registers auto-cleanup, so unmount between tests —
// otherwise a second render() leaks the first editor and getByTestId('editor')
// matches multiple nodes.
afterEach(cleanup)

// Faithful mirror of index.tsx's handlePaste *text* branches (image blobs,
// haptics, draft flush and the data-URL guard elided), driven through REAL DOM
// paste events on a contentEditable.
//
// Interaction repro for the PR #64459 review: a pasted image *path* must probe
// the host clipboard with the narrow wslHostOnly option only. Outside WSL the
// main process answers '' for that probe without reading Electron's native
// clipboard (see electron/wsl-clipboard-image.test.ts), so an unrelated
// clipboard image can never replace the pasted path text — the composer sees
// `false` and re-inserts the text unchanged. Only a WSL host bitmap (probe →
// true) may claim the paste. The empty-paste fallback keeps the pre-existing
// general probe.
type PasteClipboardImage = (opts?: { silent?: boolean; wslHostOnly?: boolean }) => Promise<boolean> | void

function Harness({ onPasteClipboardImage }: { onPasteClipboardImage?: PasteClipboardImage }) {
  const handlePaste = (event: ClipboardEvent<HTMLDivElement>) => {
    const pastedText = event.clipboardData.getData('text').trim()

    if (shouldTryHostClipboardImage(pastedText) && onPasteClipboardImage) {
      event.preventDefault()

      const editor = event.currentTarget

      const insertFallbackText = () => {
        insertPlainTextAtCaret(editor, pastedText)
      }

      void Promise.resolve(onPasteClipboardImage({ silent: true, wslHostOnly: true }))
        .then(attached => {
          if (!attached) {
            insertFallbackText()
          }
        })
        .catch(insertFallbackText)

      return
    }

    if (!pastedText) {
      event.preventDefault()

      if (onPasteClipboardImage) {
        void onPasteClipboardImage({ silent: true })
      }

      return
    }

    event.preventDefault()
    insertPlainTextAtCaret(event.currentTarget, pastedText)
  }

  return <div contentEditable data-testid="editor" onPaste={handlePaste} suppressContentEditableWarning />
}

function pasteText(editor: HTMLElement, text: string) {
  fireEvent.paste(editor, {
    clipboardData: { getData: () => text } as unknown as DataTransfer
  })
}

// The probe's .then/.catch settle on the microtask queue; a macrotask hop
// guarantees they ran before asserting the editor stayed untouched.
const flushPasteProbe = () => new Promise<void>(resolve => setTimeout(resolve, 0))

describe('composer paste: image-path host clipboard probe', () => {
  it('attaches the WSL host bitmap for a pasted image path and suppresses the text (WSL success)', async () => {
    const probe = vi.fn<PasteClipboardImage>(async () => true)
    const editor = render(<Harness onPasteClipboardImage={probe} />).getByTestId('editor')

    pasteText(editor, '/tmp/screenshot_20260714_155955_245.png')

    expect(probe).toHaveBeenCalledExactlyOnceWith({ silent: true, wslHostOnly: true })
    await flushPasteProbe()
    expect(editor.textContent).toBe('')
  })

  it('re-inserts the pasted path text when the WSL host has no image (WSL failure)', async () => {
    const probe = vi.fn<PasteClipboardImage>(async () => false)
    const editor = render(<Harness onPasteClipboardImage={probe} />).getByTestId('editor')

    pasteText(editor, '/tmp/report.png')

    expect(probe).toHaveBeenCalledExactlyOnceWith({ silent: true, wslHostOnly: true })
    await waitFor(() => expect(editor.textContent).toBe('/tmp/report.png'))
  })

  it('preserves image-path text outside WSL, where the wslHostOnly probe resolves empty', async () => {
    // Mirrors pasteClipboardImage over the fixed IPC: off WSL, saveClipboardImage
    // with wslHostOnly returns '' even with an unrelated image on the native
    // clipboard, so the renderer resolves false and the path text survives.
    const probe = vi.fn<PasteClipboardImage>(async () => false)
    const editor = render(<Harness onPasteClipboardImage={probe} />).getByTestId('editor')

    pasteText(editor, '/tmp/report.png')

    expect(probe).toHaveBeenCalledExactlyOnceWith({ silent: true, wslHostOnly: true })
    await waitFor(() => expect(editor.textContent).toBe('/tmp/report.png'))
  })

  it('keeps ordinary pasted text on the plain text path without probing the clipboard', () => {
    const probe = vi.fn<PasteClipboardImage>(async () => true)
    const editor = render(<Harness onPasteClipboardImage={probe} />).getByTestId('editor')

    pasteText(editor, 'spiegami questa immagine')

    expect(probe).not.toHaveBeenCalled()
    expect(editor.textContent).toBe('spiegami questa immagine')
  })

  it('keeps the empty-paste fallback on the general clipboard probe (no wslHostOnly)', async () => {
    const probe = vi.fn<PasteClipboardImage>(async () => false)
    const editor = render(<Harness onPasteClipboardImage={probe} />).getByTestId('editor')

    pasteText(editor, '')

    expect(probe).toHaveBeenCalledExactlyOnceWith({ silent: true })
    await flushPasteProbe()
    expect(editor.textContent).toBe('')
  })
})
