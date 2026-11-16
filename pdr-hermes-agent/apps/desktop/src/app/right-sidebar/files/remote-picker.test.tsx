import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopFsRemotePicker } from '@/lib/desktop-fs'

import { RemoteFolderPicker } from './remote-picker'

// A tiny backend filesystem: the picker reads it through readDesktopDir and
// grows it through createRemoteDir, the same two calls it makes for real.
const dirs = vi.hoisted(() => new Set<string>())
const picker = vi.hoisted(() => ({ current: null as DesktopFsRemotePicker | null }))

const createRemoteDir = vi.hoisted(() =>
  vi.fn(async (path: string) => {
    dirs.add(path)

    return path
  })
)

vi.mock('@/lib/desktop-fs', () => ({
  createRemoteDir,
  readDesktopDir: async (path: string) => {
    const prefix = path === '/' ? '/' : `${path}/`

    const entries = [...dirs]
      .filter(dir => dir.startsWith(prefix) && !dir.slice(prefix.length).includes('/') && dir !== path)
      .map(dir => ({ isDirectory: true, name: dir.slice(prefix.length), path: dir }))

    return { entries }
  },
  setDesktopFsRemotePicker: (next: DesktopFsRemotePicker | null) => {
    picker.current = next
  }
}))

function openPicker(defaultPath: string) {
  let result: Promise<string[]> = Promise.resolve([])

  act(() => {
    result = picker.current!.selectPaths({ defaultPath, directories: true })
  })

  return result
}

beforeEach(() => {
  dirs.clear()
  dirs.add('/home')
  dirs.add('/home/me')
  dirs.add('/home/me/existing')
  createRemoteDir.mockClear()
})

afterEach(cleanup)

describe('RemoteFolderPicker', () => {
  it('creates a folder on the backend under the current path and selects it', async () => {
    render(<RemoteFolderPicker />)
    const selection = openPicker('/home/me')

    await screen.findByText('existing')

    fireEvent.click(screen.getByRole('button', { name: 'New folder' }))
    const input = screen.getByRole('textbox', { name: 'Folder name' })
    fireEvent.change(input, { target: { value: 'fresh-project' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    await waitFor(() => expect(createRemoteDir).toHaveBeenCalledWith('/home/me/fresh-project'))

    // The picker lands inside the new folder, so Select picks it.
    await waitFor(() => expect(screen.queryByRole('textbox', { name: 'Folder name' })).toBeNull())
    fireEvent.click(screen.getByRole('button', { name: 'Select folder' }))

    await expect(selection).resolves.toEqual(['/home/me/fresh-project'])
  })

  it('rejects names that would escape the current folder without calling the backend', async () => {
    render(<RemoteFolderPicker />)
    void openPicker('/home/me')

    await screen.findByText('existing')

    fireEvent.click(screen.getByRole('button', { name: 'New folder' }))
    const input = screen.getByRole('textbox', { name: 'Folder name' })

    for (const name of ['..', 'a/b']) {
      fireEvent.change(input, { target: { value: name } })
      fireEvent.keyDown(input, { key: 'Enter' })
    }

    expect(createRemoteDir).not.toHaveBeenCalled()
    expect(screen.getByRole('textbox', { name: 'Folder name' })).toBeTruthy()
  })

  it('Escape abandons the new folder name without closing the picker', async () => {
    render(<RemoteFolderPicker />)
    void openPicker('/home/me')

    await screen.findByText('existing')

    fireEvent.click(screen.getByRole('button', { name: 'New folder' }))
    fireEvent.keyDown(screen.getByRole('textbox', { name: 'Folder name' }), { key: 'Escape' })

    expect(screen.queryByRole('textbox', { name: 'Folder name' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Select folder' })).toBeTruthy()
    expect(createRemoteDir).not.toHaveBeenCalled()
  })

  it('surfaces a backend failure inline and keeps the picker where it was', async () => {
    createRemoteDir.mockRejectedValueOnce(new Error('Directory is not writable'))
    render(<RemoteFolderPicker />)
    const selection = openPicker('/home/me')

    await screen.findByText('existing')

    fireEvent.click(screen.getByRole('button', { name: 'New folder' }))
    const input = screen.getByRole('textbox', { name: 'Folder name' })
    fireEvent.change(input, { target: { value: 'locked' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    await screen.findByText(/Directory is not writable/)
    fireEvent.click(screen.getByRole('button', { name: 'Select folder' }))

    await expect(selection).resolves.toEqual(['/home/me'])
  })
})
