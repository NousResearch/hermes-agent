import { afterEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  copyTextToClipboard: vi.fn(async () => undefined),
  isDesktopFsRemoteMode: vi.fn(() => false),
  notify: vi.fn(),
  notifyError: vi.fn(),
  notifyWorkspaceChanged: vi.fn(),
  renameDesktopPath: vi.fn(async () => undefined),
  revealDesktopPath: vi.fn(async () => undefined),
  trashDesktopPath: vi.fn(async () => undefined)
}))

vi.mock('@/i18n', () => ({
  translateNow: (key: string) => key
}))

vi.mock('@/lib/desktop-fs', () => ({
  copyTextToClipboard: mocks.copyTextToClipboard,
  isDesktopFsRemoteMode: mocks.isDesktopFsRemoteMode,
  renameDesktopPath: mocks.renameDesktopPath,
  revealDesktopPath: mocks.revealDesktopPath,
  trashDesktopPath: mocks.trashDesktopPath
}))

vi.mock('@/store/notifications', () => ({
  notify: mocks.notify,
  notifyError: mocks.notifyError
}))

vi.mock('@/store/workspace-events', () => ({
  notifyWorkspaceChanged: mocks.notifyWorkspaceChanged
}))

import {
  $fileActionDialog,
  $renamingPath,
  beginInlineRename,
  executeFileDelete,
  executeFileRename,
  requestFileDelete,
  revealFile
} from './file-actions'

describe('file action remote-mode guard', () => {
  afterEach(() => {
    vi.clearAllMocks()
    mocks.isDesktopFsRemoteMode.mockReturnValue(false)
    $fileActionDialog.set(null)
    $renamingPath.set(null)
  })

  it('blocks reveal/rename/delete entry points for remote gateway files', async () => {
    mocks.isDesktopFsRemoteMode.mockReturnValue(true)

    beginInlineRename('/remote/repo/file.ts')
    requestFileDelete({ isDirectory: false, name: 'file.ts', path: '/remote/repo/file.ts' })
    await revealFile('/remote/repo/file.ts')
    await executeFileRename('/remote/repo/file.ts', 'next.ts')
    await executeFileDelete('/remote/repo/file.ts')

    expect($renamingPath.get()).toBeNull()
    expect($fileActionDialog.get()).toBeNull()
    expect(mocks.revealDesktopPath).not.toHaveBeenCalled()
    expect(mocks.renameDesktopPath).not.toHaveBeenCalled()
    expect(mocks.trashDesktopPath).not.toHaveBeenCalled()
    expect(mocks.notifyWorkspaceChanged).not.toHaveBeenCalled()
    expect(mocks.notify).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'warning',
        message: expect.stringContaining('This file lives on the gateway')
      })
    )
    expect(mocks.notify).toHaveBeenCalledWith(
      expect.objectContaining({ message: expect.stringContaining('not available for a remote file') })
    )
  })

  it('keeps local file actions unchanged', async () => {
    mocks.isDesktopFsRemoteMode.mockReturnValue(false)

    beginInlineRename('/local/repo/file.ts')
    requestFileDelete({ isDirectory: false, name: 'file.ts', path: '/local/repo/file.ts' })
    await revealFile('/local/repo/file.ts')
    await executeFileRename('/local/repo/file.ts', 'next.ts')
    await executeFileDelete('/local/repo/file.ts')

    expect($renamingPath.get()).toBe('/local/repo/file.ts')
    expect($fileActionDialog.get()).toMatchObject({ kind: 'delete', path: '/local/repo/file.ts' })
    expect(mocks.revealDesktopPath).toHaveBeenCalledWith('/local/repo/file.ts')
    expect(mocks.renameDesktopPath).toHaveBeenCalledWith('/local/repo/file.ts', 'next.ts')
    expect(mocks.trashDesktopPath).toHaveBeenCalledWith('/local/repo/file.ts')
    expect(mocks.notifyWorkspaceChanged).toHaveBeenCalledTimes(2)
  })
})
