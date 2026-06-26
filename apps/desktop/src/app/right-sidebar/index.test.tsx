import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'
import { $connection, setCurrentCwd } from '@/store/session'

import { resetProjectTreeState } from './files/use-project-tree'

import { RightSidebarPane } from './index'

const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()

function installBridge() {
  ;(window as unknown as { hermesDesktop: { readDir: typeof readDir } }).hermesDesktop = { readDir }
}

describe('RightSidebarPane', () => {
  beforeEach(() => {
    $connection.set(null)
    resetProjectTreeState()
    readDir.mockReset()
    readDir.mockResolvedValue({ entries: [{ isDirectory: false, name: 'README.md', path: '/repo/README.md' }] })
    installBridge()
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    setCurrentCwd('')
    resetProjectTreeState()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('renders the tree and an "Open folder" button when the session has a working dir', async () => {
    setCurrentCwd('/repo')

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} onChangeCwd={vi.fn()} />)

    const refresh = await screen.findByRole('button', { name: 'Refresh tree' })

    readDir.mockClear()
    fireEvent.click(refresh)
    await waitFor(() => expect(readDir).toHaveBeenCalledWith('/repo'))

    // The folder picker is available so the user can change the working dir.
    expect(screen.getByRole('button', { name: 'Open folder' })).toBeDefined()
  })

  it('shows the folder picker (not a dead "No project open") for a detached chat', async () => {
    setCurrentCwd('')

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} onChangeCwd={vi.fn()} />)

    // No tree to refresh (no cwd yet), but the "Open folder" button is present
    // so the user can pick a folder to start working in.
    expect(screen.queryByRole('button', { name: 'Refresh tree' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Open folder' })).toBeDefined()
    expect(readDir).not.toHaveBeenCalled()
  })
})