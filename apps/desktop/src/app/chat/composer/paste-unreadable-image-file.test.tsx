// @vitest-environment jsdom
import { AssistantRuntimeProvider, useExternalStoreRuntime } from '@assistant-ui/react'
import type { ThreadMessageLike } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { RICH_INPUT_SLOT } from './rich-editor'
import type { ChatBarProps, ChatBarState } from './types'

import { ChatBar } from './index'

afterEach(cleanup)

// THE INVARIANT: a pasted image File the renderer cannot read still attaches.
//
// macOS Universal Clipboard (an image copied on an iPhone) puts a file URL on
// the pasteboard. Chromium hands the paste event a File for it, but macOS is
// still fetching that file from the phone when the paste captures it, so the
// read rejects with NotReadableError — and the composer toasted "Image attach
// failed" with nothing attached (#159184). Main can still read the clipboard's
// file URL, so the paste falls back to it.
//
// Driven through the REAL ChatBar with a real paste event on the focused
// contentEditable; jsdom has no clipboard, so the event carries the fake
// DataTransfer shape the sibling paste tests use.
const state: ChatBarState = {
  model: { canSwitch: false, model: '', provider: '' },
  tools: { enabled: false, label: '' },
  voice: { enabled: false, active: false }
}

type ImageProps = Pick<ChatBarProps, 'onAttachImageBlob' | 'onPasteClipboardImage'>

function Harness(props: ImageProps) {
  const runtime = useExternalStoreRuntime({
    convertMessage: (message: ThreadMessageLike) => message,
    isRunning: false,
    messages: [] as ThreadMessageLike[],
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <ChatBar
            busy={false}
            disabled={false}
            gateway={null}
            onCancel={vi.fn()}
            onSubmit={vi.fn(async () => true)}
            state={state}
            {...props}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  )
}

/** An image File as a file-URL paste yields it: real bytes, optionally unreadable. */
function pastedImageFile({ readable }: { readable: boolean }) {
  const file = new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47, 1, 2, 3, 4])], 'IMG_0001.png', { type: 'image/png' })

  if (!readable) {
    file.arrayBuffer = () =>
      Promise.reject(new DOMException('The requested file could not be read', 'NotReadableError'))
  }

  return file
}

function pasteFile(editor: HTMLElement, file: File) {
  Object.defineProperty(editor, 'isContentEditable', { configurable: true, value: true })
  editor.focus()

  const event = new Event('paste', { bubbles: true, cancelable: true }) as ClipboardEvent

  Object.defineProperty(event, 'clipboardData', {
    value: {
      getData: () => '',
      files: Object.assign([file], { item: (index: number) => [file][index] ?? null }),
      items: [{ getAsFile: () => file, kind: 'file', type: file.type }]
    }
  })

  act(() => {
    fireEvent(editor, event)
  })
}

function renderComposer() {
  const onAttachImageBlob = vi.fn(async () => true)
  const onPasteClipboardImage = vi.fn(async () => true)

  const { container } = render(
    <Harness onAttachImageBlob={onAttachImageBlob} onPasteClipboardImage={onPasteClipboardImage} />
  )

  return {
    editor: container.querySelector<HTMLElement>(`[data-slot="${RICH_INPUT_SLOT}"]`)!,
    onAttachImageBlob,
    onPasteClipboardImage
  }
}

describe('a pasted image file', () => {
  it('falls back to the main-process clipboard read when the renderer cannot read it', async () => {
    const { editor, onAttachImageBlob, onPasteClipboardImage } = renderComposer()

    pasteFile(editor, pastedImageFile({ readable: false }))

    await waitFor(() => expect(onPasteClipboardImage).toHaveBeenCalledTimes(1))
    expect(onAttachImageBlob).not.toHaveBeenCalled()
  })

  it('attaches its bytes in the renderer when it is readable', async () => {
    const { editor, onAttachImageBlob, onPasteClipboardImage } = renderComposer()

    pasteFile(editor, pastedImageFile({ readable: true }))

    await waitFor(() => expect(onAttachImageBlob).toHaveBeenCalledTimes(1))

    const [attached] = onAttachImageBlob.mock.calls[0] as unknown as [File]

    expect(attached.name).toBe('IMG_0001.png')
    expect(new Uint8Array(await attached.arrayBuffer())).toEqual(new Uint8Array([0x89, 0x50, 0x4e, 0x47, 1, 2, 3, 4]))
    expect(onPasteClipboardImage).not.toHaveBeenCalled()
  })
})
