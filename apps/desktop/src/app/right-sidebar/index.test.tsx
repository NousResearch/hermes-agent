import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import type { HermesReadDirResult } from '@/global'
import { $activeProjectId, $projects } from '@/store/projects'
import { $connection, $selectedStoredSessionId, $sessions, $workspaceCwdOwner, setCurrentCwd } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'

import { resetProjectTreeState } from './files/use-project-tree'

import { RightSidebarPane } from './index'

const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()
const repoStatus = vi.fn<(cwd: string) => Promise<null>>()

function installBridge() {
  ;(
    window as unknown as { hermesDesktop: { git: { repoStatus: typeof repoStatus }; readDir: typeof readDir } }
  ).hermesDesktop = {
    git: { repoStatus },
    readDir
  }
}

describe('RightSidebarPane', () => {
  beforeEach(() => {
    $activeProjectId.set(null)
    $projects.set([])
    $connection.set(null)
    $selectedStoredSessionId.set(null)
    $workspaceCwdOwner.set(null)
    resetProjectTreeState()
    readDir.mockReset()
    readDir.mockResolvedValue({ entries: [{ isDirectory: false, name: 'README.md', path: '/repo/README.md' }] })
    repoStatus.mockReset()
    repoStatus.mockResolvedValue(null)
    installBridge()
  })

  afterEach(() => {
    cleanup()
    $activeProjectId.set(null)
    $projects.set([])
    $connection.set(null)
    $selectedStoredSessionId.set(null)
    $workspaceCwdOwner.set(null)
    setCurrentCwd('')
    resetProjectTreeState()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it.each(['/repo-one', '/other-repo'])('shows project folders only for the focused workspace %s', async cwd => {
    setCurrentCwd(cwd)
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

    if (cwd === '/other-repo') {
      await screen.findByRole('button', { name: 'Refresh tree' })
      await waitFor(() => expect(readDir).toHaveBeenCalledWith(cwd))
      expect(screen.getAllByRole('button', { name: 'Refresh tree' })).toHaveLength(1)
      expect(readDir).not.toHaveBeenCalledWith('/repo-one')
      expect(readDir).not.toHaveBeenCalledWith('/repo-two')

      return
    }

    await waitFor(() => {
      expect(readDir).toHaveBeenCalledWith('/repo-one')
      expect(readDir).toHaveBeenCalledWith('/repo-two')
    })
    const refreshButtons = screen.getAllByRole('button', { name: 'Refresh tree' })
    expect(refreshButtons).toHaveLength(2)
    await waitFor(() => refreshButtons.forEach(button => expect((button as HTMLButtonElement).disabled).toBe(false)))

    readDir.mockClear()
    repoStatus.mockClear()
    refreshButtons.forEach(button => fireEvent.click(button))
    await waitFor(() => {
      expect(readDir).toHaveBeenCalledWith('/repo-one')
      expect(readDir).toHaveBeenCalledWith('/repo-two')
      expect(repoStatus).toHaveBeenCalledWith('/repo-one')
      expect(repoStatus).toHaveBeenCalledWith('/repo-two')
    })
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

  it('reads the focused tile workspace cwd when a tile tab is focused', async () => {
    $selectedStoredSessionId.set('main-session')
    $workspaceCwdOwner.set('main-session')
    setCurrentCwd('/repo-main')

    $sessions.set([{ cwd: '/repo-tile', id: 'tile-session' } as any])
    $sessionTiles.set([{ storedSessionId: 'tile-session', runtimeId: 'rt-tile', workspaceMode: 'sessions' } as any])
    $layoutTree.set({
      id: 'grp-1',
      type: 'group',
      panes: ['session-tile:tile-session'],
      active: 'session-tile:tile-session'
    } as any)
    $activeTreeGroup.set('grp-1')

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    const refresh = await screen.findByRole('button', { name: 'Refresh tree' })
    readDir.mockClear()
    repoStatus.mockClear()
    fireEvent.click(refresh)
    await waitFor(() => {
      expect(readDir).toHaveBeenCalledWith('/repo-tile')
      expect(repoStatus).toHaveBeenCalledWith('/repo-tile')
    })
  })

  it('keeps the focused tile workspace cwd even when clicking into the sidebar group', async () => {
    $selectedStoredSessionId.set('main-session')
    $workspaceCwdOwner.set('main-session')
    setCurrentCwd('/repo-main')

    $sessions.set([{ cwd: '/repo-tile', id: 'tile-session' } as any])
    $sessionTiles.set([{ storedSessionId: 'tile-session', runtimeId: 'rt-tile', workspaceMode: 'sessions' } as any])
    $layoutTree.set({
      id: 'grp-main',
      type: 'group',
      panes: ['workspace', 'session-tile:tile-session'],
      active: 'session-tile:tile-session'
    } as any)
    // User clicks into the files sidebar group:
    $activeTreeGroup.set('grp-files')

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    const refresh = await screen.findByRole('button', { name: 'Refresh tree' })
    readDir.mockClear()
    repoStatus.mockClear()
    fireEvent.click(refresh)
    await waitFor(() => {
      expect(readDir).toHaveBeenCalledWith('/repo-tile')
      expect(repoStatus).toHaveBeenCalledWith('/repo-tile')
    })
  })
})
