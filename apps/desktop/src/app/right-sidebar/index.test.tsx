import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'
import { $connection, setCurrentCwd } from '@/store/session'

import { resetProjectTreeState } from './files/use-project-tree'

import { RightSidebarPane } from './index'

const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()
const selectPaths = vi.fn()
const localFilesPolicy = vi.fn()

function ok(entries: { name: string; path: string; isDirectory: boolean }[]): HermesReadDirResult {
  return { entries }
}

function installBridge() {
  ;(
    window as unknown as {
      hermesDesktop: {
        readDir: typeof readDir
        selectPaths: typeof selectPaths
        localFilesPolicy: typeof localFilesPolicy
      }
    }
  ).hermesDesktop = { readDir, selectPaths, localFilesPolicy }
}

describe('RightSidebarPane', () => {
  beforeEach(() => {
    $connection.set(null)
    resetProjectTreeState()
    setCurrentCwd('/repo')
    readDir.mockReset()
    selectPaths.mockReset()
    localFilesPolicy.mockReset()
    readDir.mockResolvedValue(ok([{ name: 'README.md', path: '/repo/README.md', isDirectory: false }]))
    selectPaths.mockResolvedValue(['/repo-next'])
    localFilesPolicy.mockResolvedValue({ disabled: false, reason: null })
    installBridge()
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    setCurrentCwd('')
    resetProjectTreeState()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('refreshes the current tree without opening the folder picker', async () => {
    const onChangeCwd = vi.fn()

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} onChangeCwd={onChangeCwd} />)

    await waitFor(() => expect(screen.getByRole('button', { name: 'Refresh tree' }).hasAttribute('disabled')).toBe(false))

    readDir.mockClear()

    fireEvent.click(screen.getByRole('button', { name: 'Refresh tree' }))

    await waitFor(() => expect(readDir).toHaveBeenCalledWith('/repo'))
    expect(selectPaths).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Open folder' }))

    await waitFor(() =>
      expect(selectPaths).toHaveBeenCalledWith({
        defaultPath: '/repo',
        directories: true,
        multiple: false,
        title: 'Change working directory'
      })
    )
    await waitFor(() => expect(onChangeCwd).toHaveBeenCalledWith('/repo-next'))
  })

  it('blocks local file browsing in remote-only mode', async () => {
    localFilesPolicy.mockResolvedValue({
      disabled: true,
      reason: 'Remote-only mode is enabled.'
    })

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} onChangeCwd={vi.fn()} />)

    // The disabled state replaces the local tree and locks the picker.
    // getByText throws until the remote-only body renders, so waitFor settles on it.
    await waitFor(() => screen.getByText(/Local file browsing is disabled/))
    expect(screen.getByRole('button', { name: 'Open folder' }).hasAttribute('disabled')).toBe(true)

    fireEvent.click(screen.getByRole('button', { name: 'Open folder' }))
    expect(selectPaths).not.toHaveBeenCalled()
  })
})
