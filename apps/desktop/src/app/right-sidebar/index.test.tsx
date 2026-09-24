import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'
import { $connection, $selectedStoredSessionId, $workspaceCwdOwner, setCurrentCwd } from '@/store/session'

import { $showIgnoredRoots } from './files/prefs'
import { resetProjectTreeState } from './files/use-project-tree'
import { RightSidebarPane } from './index'

vi.mock('@/api/client', () => ({ hermesApi: vi.fn().mockResolvedValue({ entries: [] }) }))

const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()
function installBridge() {
  ;(window as unknown as { hermesDesktop: { readDir: typeof readDir } }).hermesDesktop = { readDir }
}

describe('RightSidebarPane', () => {
  beforeEach(() => {
    $connection.set(null)
    $selectedStoredSessionId.set(null)
    $workspaceCwdOwner.set(null)
    $showIgnoredRoots.set([])
    resetProjectTreeState()
    readDir.mockReset()
    readDir.mockResolvedValue({ entries: [{ isDirectory: false, name: 'README.md', path: '/repo/README.md' }] })
    installBridge()
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    $selectedStoredSessionId.set(null)
    $workspaceCwdOwner.set(null)
    $showIgnoredRoots.set([])
    setCurrentCwd('')
    resetProjectTreeState()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('renders the tree whenever the session has a working dir (repo or not) — no picker', async () => {
    setCurrentCwd('/repo')

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    const refresh = await screen.findByRole('button', { name: 'Refresh tree' })
    readDir.mockClear()
    fireEvent.click(refresh)
    await waitFor(() => expect(readDir).toHaveBeenCalledWith('/repo'))

    // The freeform folder picker is retired.
    expect(screen.queryByRole('button', { name: 'Open folder' })).toBeNull()
  })

  it('does not read a retained cwd while it belongs to a previous session', async () => {
    $selectedStoredSessionId.set('new-session')
    $workspaceCwdOwner.set('previous-session')
    setCurrentCwd('/home/doug/default-profile-workspace')

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Refresh tree' })).toBeNull())
    expect(readDir).not.toHaveBeenCalled()
  })

  it('shows no tree for a detached chat (no working dir)', async () => {
    setCurrentCwd('')

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Refresh tree' })).toBeNull())
    expect(readDir).not.toHaveBeenCalled()
  })

  it('explains the remote security filter when showing gitignored files', async () => {
    $connection.set({ mode: 'remote' } as never)
    setCurrentCwd('/repo')
    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    const notice = /remote backends hide sensitive files/i
    expect(screen.queryByText(notice)).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Show gitignored files' }))
    expect(await screen.findByText(notice)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Hide gitignored files' }))
    expect(screen.queryByText(notice)).toBeNull()
    expect(readDir).not.toHaveBeenCalled()
  })

  it('does not claim local files are hidden by the remote security filter', () => {
    setCurrentCwd('/repo')
    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Show gitignored files' }))
    expect(screen.queryByText(/remote backends hide sensitive files/i)).toBeNull()
  })
})
