import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { selectDesktopPaths } from '@/lib/desktop-fs'
import { $connection } from '@/store/session'

import { RemoteFolderPicker } from './remote-picker'

// The picker drives the real desktop-fs facade against a stubbed Electron
// bridge, so the test covers the whole loop: dialog open -> list -> create ->
// POST /api/fs/mkdir -> land inside the new folder -> resolve the selection.
const api = vi.fn(
  async (request: { body?: { path?: string }; method?: string; path: string; profile?: string }) => {
    if (request.path.startsWith('/api/fs/list?')) {
      const dir = decodeURIComponent(request.path.slice(request.path.indexOf('path=') + 'path='.length))

      if (dir === '/remote') {
        return { entries: [{ name: 'existing', path: '/remote/existing', isDirectory: true }] }
      }

      return { entries: [] }
    }

    if (request.path === '/api/fs/mkdir') {
      return { ok: true, path: request.body?.path }
    }

    throw new Error(`unexpected path ${request.path}`)
  }
)

function openPicker(defaultPath = '/remote') {
  let selection!: Promise<string[]>
  act(() => {
    selection = selectDesktopPaths({ defaultPath, directories: true })
  })

  return selection
}

describe('RemoteFolderPicker new folder', () => {
  beforeEach(() => {
    $connection.set({ mode: 'remote' } as never)
    ;(window as unknown as Record<string, unknown>).hermesDesktop = { api }
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <RemoteFolderPicker />
      </I18nProvider>
    )
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    delete (window as unknown as Record<string, unknown>).hermesDesktop
    vi.clearAllMocks()
  })

  it('creates a folder on the backend and selects it', async () => {
    const selection = openPicker()

    // The listing effect runs inside act() above; wait for the real row.
    await screen.findByRole('button', { name: 'existing' })

    fireEvent.click(screen.getByRole('button', { name: 'New folder' }))

    const input = screen.getByPlaceholderText('Folder name')
    fireEvent.change(input, { target: { value: 'fresh dir' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    await waitFor(() =>
      expect(api).toHaveBeenCalledWith(
        expect.objectContaining({ body: { path: '/remote/fresh dir' }, method: 'POST', path: '/api/fs/mkdir' })
      )
    )

    // After creation the picker lands inside the new folder, ready to select.
    await screen.findByRole('button', { name: 'fresh dir' })

    fireEvent.click(screen.getByRole('button', { name: 'Select folder' }))

    await expect(selection).resolves.toEqual(['/remote/fresh dir'])
  })

  it('surfaces a backend failure inline and stays on the current folder', async () => {
    const selection = openPicker()

    await screen.findByRole('button', { name: 'existing' })
    fireEvent.click(screen.getByRole('button', { name: 'New folder' }))

    // Reject the next backend call — which is the mkdir POST, not the listing.
    api.mockRejectedValueOnce(new Error('409: {"detail":"Path already exists"}'))

    const input = screen.getByPlaceholderText('Folder name')
    fireEvent.change(input, { target: { value: 'existing' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    await screen.findByText(/Could not create the folder/)
    expect(screen.getByText(/Path already exists/)).toBeTruthy()

    // The dialog is still open on /remote; cancelling resolves with no selection.
    // (The footer Cancel is the text button — the inline editor's icon-only
    // cancel shares the accessible name while the editor is open.)
    fireEvent.click(screen.getByText('Cancel'))
    await expect(selection).resolves.toEqual([])
  })

  it('rejects names containing a slash before any backend call', async () => {
    openPicker()

    await screen.findByRole('button', { name: 'existing' })
    fireEvent.click(screen.getByRole('button', { name: 'New folder' }))

    const input = screen.getByPlaceholderText('Folder name')
    fireEvent.change(input, { target: { value: 'a/b' } })

    expect(screen.getByRole('button', { name: 'Confirm' })).toHaveProperty('disabled', true)

    fireEvent.keyDown(input, { key: 'Enter' })
    expect(api).not.toHaveBeenCalledWith(expect.objectContaining({ path: '/api/fs/mkdir' }))
  })

  it('drops a half-typed name when navigating away', async () => {
    openPicker()

    await screen.findByRole('button', { name: 'existing' })
    fireEvent.click(screen.getByRole('button', { name: 'New folder' }))
    fireEvent.change(screen.getByPlaceholderText('Folder name'), { target: { value: 'draft' } })

    fireEvent.click(screen.getByRole('button', { name: 'existing' }))

    // Navigating into a subfolder closed the inline editor with the draft.
    await waitFor(() => expect(screen.queryByPlaceholderText('Folder name')).toBeNull())
  })
})
