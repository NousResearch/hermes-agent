import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const fs = vi.hoisted(() => ({
  create: vi.fn(),
  rename: vi.fn(),
  trash: vi.fn()
}))

vi.mock('@/lib/desktop-fs', () => ({
  captureDesktopFsOperationScope: () => ({ mode: 'remote', cacheKey: 'test' }),
  desktopFsCacheKey: () => 'test',
  isDesktopFsRemoteMode: () => true,
  createDesktopEntry: fs.create,
  renameDesktopPath: fs.rename,
  trashDesktopPath: fs.trash,
  copyTextToClipboard: vi.fn(),
  revealDesktopPath: vi.fn()
}))

import {
  $creatingEntry, $fileActionDialog, $renamingPath, beginInlineRename, cancelInlineRename,
  requestFileDelete, requestNewEntry
} from '@/store/file-actions'

import { FileActionDialogs, InlineRenameInput } from './file-actions'

function deferred() {
  let resolve!: () => void
  const promise = new Promise<void>(r => { resolve = r })

  return { promise, resolve }
}

afterEach(() => {
  cleanup()
  $creatingEntry.set(null)
  $fileActionDialog.set(null)
  cancelInlineRename()
  vi.resetAllMocks()
})

describe('file action editor ownership', () => {
  it('does not let an old rename clear a newer rename', async () => {
    const pending = deferred()
    fs.rename.mockReturnValue(pending.promise)
    beginInlineRename('/work/a')
    render(<InlineRenameInput name="a" path="/work/a" />)
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'renamed' } })
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' })
    act(() => beginInlineRename('/work/b'))
    await act(async () => pending.resolve())
    expect($renamingPath.get()).toBe('/work/b')
  })

  it('ignores a stale create editor instead of cancelling its replacement', async () => {
    await requestNewEntry({ directory: false, parentDir: '/work' })
    render(<InlineRenameInput name="" path="/work" />)
    const oldInput = screen.getByRole('textbox')
    await act(async () => requestNewEntry({ directory: true, parentDir: '/work' }))
    const replacement = $creatingEntry.get()
    expect(screen.getByRole('textbox')).not.toBe(oldInput)
    fireEvent.keyDown(oldInput, { key: 'Escape' })
    expect($creatingEntry.get()).toBe(replacement)
    expect(fs.create).not.toHaveBeenCalled()
  })

  it('passes unsafe names unchanged to the filesystem validator', async () => {
    fs.create.mockRejectedValue(new Error('invalid name'))
    await requestNewEntry({ directory: false, parentDir: '/work' })
    render(<InlineRenameInput name="" path="/work" />)
    fireEvent.change(screen.getByRole('textbox'), { target: { value: '../other' } })
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' })
    await waitFor(() => expect(fs.create).toHaveBeenCalledWith('/work', '../other', false, expect.anything()))
  })

  it('keeps the replacement delete dialog when an older deletion resolves', async () => {
    const pending = deferred()
    fs.trash.mockReturnValue(pending.promise)
    requestFileDelete({ isDirectory: false, name: 'a', path: '/work/a' })
    render(<FileActionDialogs />)
    fireEvent.click(screen.getByRole('button', { name: /delete/i }))
    act(() => requestFileDelete({ isDirectory: false, name: 'b', path: '/work/b' }))
    await act(async () => pending.resolve())
    expect($fileActionDialog.get()?.path).toBe('/work/b')
    expect(screen.getByRole('dialog').textContent).toContain('b')
  })
})
