import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'
import { $activeTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import { $connection, $selectedStoredSessionId, $sessions, $workspaceCwdOwner, setCurrentCwd } from '@/store/session'
import { $sessionStates, $sessionTiles } from '@/store/session-states'

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

  it('reads the focused tile workspace cwd when a tile tab is focused', async () => {
    $selectedStoredSessionId.set('main-session')
    $workspaceCwdOwner.set('main-session')
    setCurrentCwd('/repo-main')

    $sessions.set([
      { cwd: '/repo-tile', id: 'tile-session' } as any
    ])
    $sessionTiles.set([
      { storedSessionId: 'tile-session', runtimeId: 'rt-tile', workspaceMode: 'sessions' } as any
    ])
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
    fireEvent.click(refresh)
    await waitFor(() => expect(readDir).toHaveBeenCalledWith('/repo-tile'))
  })

  it('keeps the focused tile workspace cwd even when clicking into the sidebar group', async () => {
    $selectedStoredSessionId.set('main-session')
    $workspaceCwdOwner.set('main-session')
    setCurrentCwd('/repo-main')

    $sessions.set([
      { cwd: '/repo-tile', id: 'tile-session' } as any
    ])
    $sessionTiles.set([
      { storedSessionId: 'tile-session', runtimeId: 'rt-tile', workspaceMode: 'sessions' } as any
    ])
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
    fireEvent.click(refresh)
    await waitFor(() => expect(readDir).toHaveBeenCalledWith('/repo-tile'))
  })
})
