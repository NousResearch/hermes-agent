import { useStore } from '@nanostores/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $renamingPath, beginInlineRename, cancelInlineRename, closeFileActionDialog } from '@/store/file-actions'
import { $connection } from '@/store/session'

import { FileActionDialogs, FileEntryContextMenu, InlineRenameInput } from './file-actions'

const api = vi.fn(async () => ({ ok: true, path: '/srv/archive/a.txt' }))

function renderRemoteFileMenu() {
  render(
    <>
      <FileEntryContextMenu isDirectory={false} name="a.txt" path="/srv/project/a.txt" relativeTo="/srv/project">
        <button type="button">a.txt</button>
      </FileEntryContextMenu>
      <FileActionDialogs />
    </>
  )
}

function RenameHarness() {
  const renamingPath = useStore($renamingPath)

  return renamingPath ? <InlineRenameInput name="a.txt" path={renamingPath} /> : null
}

describe('file workspace mutation menus', () => {
  beforeEach(() => {
    $connection.set({ mode: 'remote', profile: 'prod' } as never)
    cancelInlineRename()
    closeFileActionDialog()
    api.mockClear()
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { api, writeClipboard: vi.fn() }
  })

  afterEach(() => {
    cleanup()
    cancelInlineRename()
    closeFileActionDialog()
    $connection.set(null)
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('offers rename, move, and delete for authenticated remote entries', async () => {
    renderRemoteFileMenu()
    fireEvent.contextMenu(screen.getByRole('button', { name: 'a.txt' }))

    expect(await screen.findByText('Rename…')).toBeTruthy()
    expect(screen.getByText('Move…')).toBeTruthy()
    expect(screen.getByText('Delete')).toBeTruthy()
  })

  it('moves one remote entry with its browser-root guard and item-scoped dialog', async () => {
    renderRemoteFileMenu()
    fireEvent.contextMenu(screen.getByRole('button', { name: 'a.txt' }))
    fireEvent.click(await screen.findByText('Move…'))

    expect(screen.getByText('/srv/project/a.txt')).toBeTruthy()
    expect((screen.getByRole('button', { name: 'Move' }) as HTMLButtonElement).disabled).toBe(true)
    const destination = await screen.findByRole('textbox', { name: 'Destination folder' })
    fireEvent.change(destination, { target: { value: '/srv/archive' } })
    fireEvent.click(screen.getByRole('button', { name: 'Move' }))

    await waitFor(() =>
      expect(api).toHaveBeenCalledWith({
        body: {
          browserRoot: '/srv/project',
          destination: '/srv/archive',
          source: '/srv/project/a.txt'
        },
        method: 'POST',
        path: '/api/fs/move',
        profile: 'prod'
      })
    )
  })

  it('deletes one remote entry with the visible browser root', async () => {
    renderRemoteFileMenu()
    fireEvent.contextMenu(screen.getByRole('button', { name: 'a.txt' }))
    fireEvent.click(await screen.findByText('Delete'))
    expect(screen.getByText(/permanently deletes the item on the remote machine/i)).toBeTruthy()
    fireEvent.click(await screen.findByRole('button', { name: 'Delete' }))

    await waitFor(() =>
      expect(api).toHaveBeenCalledWith({
        body: { browserRoot: '/srv/project', path: '/srv/project/a.txt' },
        method: 'POST',
        path: '/api/fs/delete',
        profile: 'prod'
      })
    )
  })

  it('keeps a failed move in its item dialog for correction and retry', async () => {
    api.mockRejectedValueOnce(new Error('Destination already exists'))
    renderRemoteFileMenu()
    fireEvent.contextMenu(screen.getByRole('button', { name: 'a.txt' }))
    fireEvent.click(await screen.findByText('Move…'))

    const destination = await screen.findByRole('textbox', { name: 'Destination folder' })
    fireEvent.change(destination, { target: { value: '/srv/archive' } })
    fireEvent.click(screen.getByRole('button', { name: 'Move' }))

    expect((await screen.findByRole('alert')).textContent).toContain('Destination already exists')
    expect(screen.getByText('/srv/project/a.txt')).toBeTruthy()
    expect((destination as HTMLInputElement).value).toBe('/srv/archive')
  })

  it('keeps a failed inline rename editable and allows retry', async () => {
    api.mockRejectedValueOnce(new Error('Name already exists'))
    beginInlineRename('/srv/project/a.txt')
    render(<RenameHarness />)

    const input = screen.getByRole('textbox')
    fireEvent.change(input, { target: { value: 'b.txt' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect((await screen.findByRole('alert')).textContent).toContain('Name already exists')
    expect(screen.getByRole('textbox')).toBeTruthy()

    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' })

    await waitFor(() => expect(screen.queryByRole('textbox')).toBeNull())
    expect(api).toHaveBeenCalledTimes(2)
  })
})
