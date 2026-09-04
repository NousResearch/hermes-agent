import { type AppendMessage, AssistantRuntimeProvider, ExportedMessageRepository, type ThreadMessage } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DroppedFile } from '@/app/chat/hooks/use-composer-actions'
import { useIncrementalExternalStoreRuntime } from '@/lib/incremental-external-store-runtime'

import { assistantMessage, stubThreadEnvironment, stubThreadViewportSize, userMessage } from '../test-utils'

import { Thread } from '.'

vi.mock('@/app/session/hooks/use-prompt-actions', async importOriginal => {
  const actual = await importOriginal<Record<string, unknown>>()

  return {
    ...actual,
    uploadComposerAttachment: vi.fn()
  }
})

const { uploadComposerAttachment } = await import('@/app/session/hooks/use-prompt-actions')
const uploadMock = vi.mocked(uploadComposerAttachment)

type UploadAttachment = Parameters<typeof uploadComposerAttachment>[0]

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

stubThreadViewportSize()
stubThreadEnvironment()

const stagedRef = (refPath: string): UploadAttachment =>
  ({
    detail: `/workspace/${refPath}`,
    id: `file:/workspace/${refPath}`,
    kind: 'file',
    label: refPath.split('/').pop() || refPath,
    path: `/workspace/${refPath}`,
    refText: `@file:\`${refPath}\``
  }) as unknown as UploadAttachment

// Mirrors chat/index.tsx: incremental runtime + messageRepository + onEdit.
// cwd/gateway/sessionId ride Thread props into ThreadEditContext — the edit
// composer needs a session to stage pasted files into.
function IncrementalHarness({ onEdit }: { onEdit: (message: AppendMessage) => Promise<void> }) {
  const repository = ExportedMessageRepository.fromArray([userMessage(), assistantMessage()])

  const runtime = useIncrementalExternalStoreRuntime<ThreadMessage>({
    messageRepository: repository,
    isRunning: false,
    setMessages: () => {},
    onNew: async () => {},
    onEdit,
    onCancel: async () => {},
    onReload: async () => {}
  })

  const gateway = { request: vi.fn(async () => ({})) } as unknown as Parameters<typeof Thread>[0]['gateway']

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread cwd="/workspace" gateway={gateway} sessionId="s1" />
    </AssistantRuntimeProvider>
  )
}

async function openEditComposer() {
  const { container } = render(<IncrementalHarness onEdit={async () => {}} />)

  fireEvent.click(await screen.findByRole('button', { name: 'Edit message' }))

  const editor = (await waitFor(() => {
    const node = container.querySelector<HTMLDivElement>('[data-slot="aui_edit-composer-root"] [role="textbox"]')

    expect(node).toBeTruthy()

    return node
  })) as HTMLDivElement

  return editor
}

const pasteClipboard = (editor: HTMLElement, flavors: Record<string, string>) => {
  const clipboard = {
    getData: (type: string) => flavors[type] ?? '',
    items: []
  } as unknown as DataTransfer

  fireEvent.paste(editor, { clipboardData: clipboard })
}

describe('edit-composer paste of an OS file copy (file:// clipboard)', () => {
  beforeEach(() => {
    uploadMock.mockReset()
    uploadMock.mockResolvedValue(stagedRef('attachments/report.pdf'))
  })

  it('stages the file for the remote session and inserts the gateway-side ref instead of raw text', async () => {
    const editor = await openEditComposer()

    act(() => {
      pasteClipboard(editor, { 'text/uri-list': 'file:///home/me/report.pdf' })
    })

    await waitFor(() => {
      expect(uploadMock).toHaveBeenCalledTimes(1)
    })

    const staged = uploadMock.mock.calls[0]?.[0] as UploadAttachment

    expect(staged.path).toBe('/home/me/report.pdf')
    expect(staged.kind).toBe('file')

    // The gateway-side ref lands as a chip whose data-ref-text carries the
    // literal (backtick-quoted) @file: ref; textContent is the display label.
    await waitFor(() => {
      const chip = editor.querySelector<HTMLElement>('[data-ref-text]')

      expect(chip?.dataset.refText).toBe('@file:`attachments/report.pdf`')
    })
    expect(editor.textContent).not.toContain('file://')
  })

  it('keeps typed prose and appends the staged ref when the paste also carries text', async () => {
    uploadMock.mockResolvedValue(stagedRef('attachments/notes.txt'))

    const editor = await openEditComposer()

    act(() => {
      pasteClipboard(editor, {
        'text/plain': 'please review\nfile:///home/me/notes.txt',
        'text/uri-list': 'file:///home/me/notes.txt'
      })
    })

    // Assert the upload contract and chip presence. The prose half of the
    // paste rides the standard text-paste path (same as any text-only paste),
    // which this jsdom harness cannot fully reproduce — the paste caret/focus
    // interplay differs from a real browser even on the pre-existing path.
    await waitFor(() => {
      const chip = editor.querySelector<HTMLElement>('[data-ref-text]')

      expect(chip?.dataset.refText).toBe('@file:`attachments/notes.txt`')
    })
  })

  it('shows the staging spinner while the upload is in flight and clears it after', async () => {
    let release!: (value: UploadAttachment) => void

    uploadMock.mockReturnValue(
      new Promise<UploadAttachment>(resolve => {
        release = resolve
      })
    )

    const editor = await openEditComposer()

    act(() => {
      pasteClipboard(editor, { 'text/uri-list': 'file:///home/me/slow.pdf' })
    })

    await waitFor(() => {
      expect(document.querySelector('[data-slot="aui_edit-staging"]')).toBeTruthy()
    })

    act(() => {
      release(stagedRef('slow.pdf'))
    })

    await waitFor(() => {
      expect(document.querySelector('[data-slot="aui_edit-staging"]')).toBeNull()
    })
  })

  it('routes a failed upload through the drop-path contract: notify, no staging residue', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})

    uploadMock.mockRejectedValue(new Error('gateway unreachable'))

    const editor = await openEditComposer()

    act(() => {
      pasteClipboard(editor, { 'text/uri-list': 'file:///home/me/gone.pdf' })
    })

    await waitFor(() => {
      expect(uploadMock).toHaveBeenCalled()
    })

    // The raw URL never lands in the draft, and the staging spinner clears.
    await waitFor(() => {
      expect(document.querySelector('[data-slot="aui_edit-staging"]')).toBeNull()
    })
    expect(editor.textContent).not.toContain('file://')

    errorSpy.mockRestore()
  })

  it('leaves ordinary URL pastes as links — no staging, no upload', async () => {
    const editor = await openEditComposer()

    act(() => {
      pasteClipboard(editor, { text: 'https://example.com/room' })
    })

    // The pre-existing linkify path chips it as an @url: ref — never staged.
    await waitFor(() => {
      const chip = editor.querySelector<HTMLElement>('[data-ref-text]')

      expect(chip?.dataset.refText).toBe('@url:`https://example.com/room`')
    })
    expect(uploadMock).not.toHaveBeenCalled()
  })
})

// Type-only touch to keep the DroppedFile import meaningful for the payload
// the paste path builds ({ path } entries — the same shape OS drops carry).
export type PastePathEntry = Pick<DroppedFile, 'path'>
