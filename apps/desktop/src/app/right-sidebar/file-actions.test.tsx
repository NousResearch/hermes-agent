import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $fileActionDialog, closeFileActionDialog, requestNewFolder } from '@/store/file-actions'
import { $connection } from '@/store/session'

import { FileActionDialogs, FileEntryContextMenu, WorkspaceContextMenu } from './file-actions'

afterEach(() => {
  cleanup()
  $connection.set(null)
  closeFileActionDialog()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('new folder context menus', () => {
  it('offers New Folder when a directory is right-clicked', async () => {
    render(
      <>
        <FileEntryContextMenu isDirectory name="docs" path="/repo/docs" relativeTo="/repo">
          <button type="button">docs</button>
        </FileEntryContextMenu>
        <FileActionDialogs />
      </>
    )

    fireEvent.contextMenu(screen.getByRole('button', { name: 'docs' }))
    fireEvent.click(await screen.findByText('New Folder…'))

    expect(await screen.findByRole('dialog', { name: 'New Folder' })).toBeTruthy()
    expect(screen.getByRole('textbox', { name: 'Folder name' })).toBeTruthy()
  })

  it('does not offer New Folder when a file is right-clicked', async () => {
    render(
      <FileEntryContextMenu isDirectory={false} name="README.md" path="/repo/README.md" relativeTo="/repo">
        <button type="button">README.md</button>
      </FileEntryContextMenu>
    )

    fireEvent.contextMenu(screen.getByRole('button', { name: 'README.md' }))

    expect(screen.queryByText('New Folder…')).toBeNull()
  })

  it('does not offer New Folder for a remote workspace directory', async () => {
    $connection.set({ mode: 'remote' } as never)
    render(
      <FileEntryContextMenu isDirectory name="docs" path="/repo/docs" relativeTo="/repo">
        <button type="button">docs</button>
      </FileEntryContextMenu>
    )

    fireEvent.contextMenu(screen.getByRole('button', { name: 'docs' }))

    expect(screen.queryByText('New Folder…')).toBeNull()
  })

  it('keeps a nested folder context menu targeted on that folder', async () => {
    render(
      <WorkspaceContextMenu path="/repo">
        <div>
          <FileEntryContextMenu isDirectory name="docs" path="/repo/docs" relativeTo="/repo">
            <button type="button">docs</button>
          </FileEntryContextMenu>
        </div>
      </WorkspaceContextMenu>
    )

    fireEvent.contextMenu(screen.getByRole('button', { name: 'docs' }))
    fireEvent.click(await screen.findByText('New Folder…'))

    expect($fileActionDialog.get()).toEqual({ kind: 'new-folder', parentPath: '/repo/docs' })
  })

  it('offers New Folder from the workspace background', async () => {
    render(
      <>
        <WorkspaceContextMenu path="/repo">
          <button type="button">workspace</button>
        </WorkspaceContextMenu>
        <FileActionDialogs />
      </>
    )

    fireEvent.contextMenu(screen.getByRole('button', { name: 'workspace' }))
    fireEvent.click(await screen.findByText('New Folder…'))

    expect(await screen.findByRole('dialog', { name: 'New Folder' })).toBeTruthy()
  })

  it('creates the named folder and closes the dialog', async () => {
    const createDirectory = vi.fn().mockResolvedValue({ path: '/repo/docs' })

    ;(window as unknown as { hermesDesktop: { createDirectory: typeof createDirectory } }).hermesDesktop = {
      createDirectory
    }

    requestNewFolder('/repo')
    render(<FileActionDialogs />)

    fireEvent.change(screen.getByRole('textbox', { name: 'Folder name' }), { target: { value: 'docs' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create Folder' }))

    await waitFor(() => expect(createDirectory).toHaveBeenCalledWith('/repo', 'docs'))
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'New Folder' })).toBeNull())
  })

  it('announces folder creation errors', async () => {
    const createDirectory = vi.fn().mockRejectedValue(new Error('Invalid folder name'))

    ;(window as unknown as { hermesDesktop: { createDirectory: typeof createDirectory } }).hermesDesktop = {
      createDirectory
    }

    requestNewFolder('/repo')
    render(<FileActionDialogs />)

    fireEvent.change(screen.getByRole('textbox', { name: 'Folder name' }), { target: { value: 'bad/name' } })
    fireEvent.click(screen.getByRole('button', { name: 'Create Folder' }))

    expect((await screen.findByRole('alert')).textContent).toBe('Invalid folder name')
  })
})
