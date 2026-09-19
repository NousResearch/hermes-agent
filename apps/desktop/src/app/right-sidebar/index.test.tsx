import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'
import { $activeProjectId, $projects } from '@/store/projects'
import { $connection, $selectedStoredSessionId, $workspaceCwdOwner, setCurrentCwd } from '@/store/session'

import { resetProjectTreeState } from './files/use-project-tree'

import { RightSidebarPane } from './index'

const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()

function installBridge() {
  ;(window as unknown as { hermesDesktop: { readDir: typeof readDir } }).hermesDesktop = { readDir }
}

describe('RightSidebarPane', () => {
  beforeEach(() => {
    $connection.set(null)
    $selectedStoredSessionId.set(null)
    $workspaceCwdOwner.set(null)
    $activeProjectId.set(null)
    $projects.set([])
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
    $activeProjectId.set(null)
    $projects.set([])
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

  it('shows every configured folder for the active multi-root project', async () => {
    setCurrentCwd('/repo-one')
    $activeProjectId.set('p_multi')
    $projects.set([
      {
        archived: false,
        board_slug: null,
        color: null,
        created_at: 0,
        description: null,
        folders: [
          { added_at: 0, is_primary: true, label: null, path: '/repo-one' },
          { added_at: 0, is_primary: false, label: null, path: '/repo-two' }
        ],
        icon: null,
        id: 'p_multi',
        name: 'Multi root',
        primary_path: '/repo-one',
        slug: 'multi-root'
      }
    ])

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    await waitFor(() => {
      expect(readDir).toHaveBeenCalledWith('/repo-one')
      expect(readDir).toHaveBeenCalledWith('/repo-two')
    })
    expect(screen.getAllByRole('button', { name: 'Refresh tree' })).toHaveLength(2)
  })

  it('shows no tree for a detached chat (no working dir)', async () => {
    setCurrentCwd('')

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Refresh tree' })).toBeNull())
    expect(readDir).not.toHaveBeenCalled()
  })
})
