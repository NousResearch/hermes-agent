import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { FileActionDialogs, FileEntryContextMenu } from './file-actions'

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

describe('file workspace mutation menus', () => {
  beforeEach(() => {
    $connection.set({ mode: 'remote', profile: 'prod' } as never)
    api.mockClear()
    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { api, writeClipboard: vi.fn() }
  })

  afterEach(() => {
    cleanup()
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
})
