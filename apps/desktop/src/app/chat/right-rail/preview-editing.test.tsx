import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// EDITING SAFETY contract for the file preview's spot editor:
//  - Edit exists ONLY for complete readable text (never binary/truncated);
//  - Save writes the FULL draft through writeDesktopFileText;
//  - a disk change since open surfaces Overwrite/Discard instead of clobbering;
//  - Cancel/Discard never touch disk.
// Electron/backend stay authoritative for I/O — everything here goes through
// the mocked @/lib/desktop-fs seam.
// (The clean-draft no-op guard lives on the real Save button's disabled
// state; the mocked editor here drives saveEdit directly.)

const { readDesktopFileText, writeDesktopFileText, desktopGitRoot, desktopFileDiff } = vi.hoisted(() => ({
  readDesktopFileText: vi.fn(),
  writeDesktopFileText: vi.fn(),
  desktopGitRoot: vi.fn(),
  desktopFileDiff: vi.fn()
}))

vi.mock('@/lib/desktop-fs', () => ({
  isDesktopFsRemoteMode: () => false,
  desktopFsCacheKey: () => 'local:',
  readDesktopFileText,
  writeDesktopFileText,
  desktopGitRoot,
  desktopFileDiff
}))

vi.mock('@/store/preview-edit', () => ({ setPreviewDirty: vi.fn() }))
vi.mock('@/store/workspace-events', () => ({ notifyWorkspaceChanged: vi.fn() }))

vi.mock('@/store/session', async () => {
  const { atom } = await import('nanostores')

  return { $connection: atom(null), $currentCwd: atom('/w') }
})

// The real CodeEditor drags in CodeMirror; the contract under test only needs
// "an editable surface that reports its value upward".
vi.mock('@/components/chat/code-editor', () => ({
  CodeEditor: ({
    initialValue,
    onChange,
    onSave
  }: {
    initialValue: string
    onChange: (value: string) => void
    onSave?: () => void
  }) => (
    <div>
      <textarea
        aria-label="spot-editor"
        defaultValue={initialValue}
        onChange={event => onChange(event.currentTarget.value)}
      />
      <button onClick={onSave} type="button">
        editor-save
      </button>
    </div>
  )
}))

import { I18nProvider } from '@/i18n'

import { LocalFilePreview } from './preview-file'

const PATH = '/w/notes.md'

const TARGET = {
  kind: 'file' as const,
  label: 'notes.md',
  language: 'markdown',
  path: PATH,
  previewKind: 'text' as const,
  source: PATH,
  url: `file://${PATH}`
}

const BASE_TEXT = '# notes\n\noriginal body'

function okRead(text = BASE_TEXT) {
  return { binary: false, byteSize: text.length, language: 'markdown', mimeType: 'text/markdown', text }
}

function mount(target = TARGET) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <LocalFilePreview reloadKey={0} target={target} />
    </I18nProvider>
  )
}

async function enterEdit() {
  fireEvent.click(await screen.findByRole('button', { name: /edit/i }))

  return screen.findByLabelText('spot-editor') as Promise<HTMLTextAreaElement>
}

beforeEach(() => {
  readDesktopFileText.mockResolvedValue(okRead())
  desktopGitRoot.mockResolvedValue(null)
  desktopFileDiff.mockResolvedValue('')
  writeDesktopFileText.mockResolvedValue({ path: PATH })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('LocalFilePreview spot editing', () => {
  it('offers Edit for complete readable text and seeds the editor with disk content', async () => {
    mount()

    const edit = await screen.findByRole('button', { name: /edit/i })
    fireEvent.click(edit)

    const editor = (await screen.findByLabelText('spot-editor')) as HTMLTextAreaElement
    expect(editor.value).toBe(BASE_TEXT)
  })

  it('does NOT offer Edit for binary content', async () => {
    readDesktopFileText.mockResolvedValue({ ...okRead(), binary: true })

    mount()

    await waitFor(() => expect(readDesktopFileText).toHaveBeenCalled())
    expect(screen.queryByRole('button', { name: /edit/i })).toBeNull()
  })

  it('does NOT offer Edit for truncated reads (saving would drop the tail)', async () => {
    readDesktopFileText.mockResolvedValue({ ...okRead(), text: `${BASE_TEXT}\n`, truncated: true })

    mount()

    await waitFor(() => expect(readDesktopFileText).toHaveBeenCalled())
    expect(screen.queryByRole('button', { name: /edit/i })).toBeNull()
  })

  it('Cancel discards the draft without writing to disk', async () => {
    mount()
    fireEvent.click(await screen.findByRole('button', { name: /edit/i }))

    const editor = (await screen.findByLabelText('spot-editor')) as HTMLTextAreaElement
    fireEvent.change(editor, { target: { value: '# rewritten' } })
    fireEvent.click(await screen.findByRole('button', { name: /^cancel$/i }))

    expect(writeDesktopFileText).not.toHaveBeenCalled()
    // Back in read view with the ORIGINAL disk content.
    expect(screen.getByText(/original body/)).toBeTruthy()
  })

  it('Save writes the full edited draft through the fs bridge and exits edit mode', async () => {
    // Read sequence: initial load, then the save-time staleness re-read (both
    // see the original), then the post-save self-reload sees the new content.
    readDesktopFileText
      .mockResolvedValueOnce(okRead())
      .mockResolvedValueOnce(okRead())
      .mockResolvedValue(okRead('# notes\n\nedited body'))

    mount()
    fireEvent.click(await screen.findByRole('button', { name: /edit/i }))

    const editor = (await screen.findByLabelText('spot-editor')) as HTMLTextAreaElement
    fireEvent.change(editor, { target: { value: '# notes\n\nedited body' } })
    fireEvent.click(screen.getByRole('button', { name: 'editor-save' }))

    await waitFor(() => expect(writeDesktopFileText).toHaveBeenCalledTimes(1))
    expect(writeDesktopFileText).toHaveBeenCalledWith(PATH, '# notes\n\nedited body')
    expect(await screen.findByText(/edited body/)).toBeTruthy()
  })

  it('surfaces Overwrite/Discard when the file changed on disk mid-edit; Discard never writes', async () => {
    mount()
    fireEvent.click(await screen.findByRole('button', { name: /edit/i }))

    // Something else edits the file while the user types.
    readDesktopFileText.mockResolvedValue(okRead('# clobbered externally'))
    const editor = (await screen.findByLabelText('spot-editor')) as HTMLTextAreaElement
    fireEvent.change(editor, { target: { value: '# my draft' } })
    fireEvent.click(screen.getByRole('button', { name: 'editor-save' }))

    // Conflict banner appears INSTEAD of a silent write.
    expect(await screen.findByText(/file changed on disk/i)).toBeTruthy()
    expect(writeDesktopFileText).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /discard & reload/i }))
    await waitFor(() => expect(screen.getByText(/clobbered externally/)).toBeTruthy())
    expect(writeDesktopFileText).not.toHaveBeenCalled()
  })

  it('Overwrite intentionally wins the conflict and persists the draft', async () => {
    mount()
    fireEvent.click(await screen.findByRole('button', { name: /edit/i }))

    readDesktopFileText.mockResolvedValue(okRead('# clobbered externally'))
    const editor = (await screen.findByLabelText('spot-editor')) as HTMLTextAreaElement
    fireEvent.change(editor, { target: { value: '# my draft' } })
    fireEvent.click(screen.getByRole('button', { name: 'editor-save' }))
    await screen.findByText(/file changed on disk/i)

    fireEvent.click(screen.getByRole('button', { name: /overwrite/i }))

    await waitFor(() => expect(writeDesktopFileText).toHaveBeenCalledWith(PATH, '# my draft'))
  })
})
