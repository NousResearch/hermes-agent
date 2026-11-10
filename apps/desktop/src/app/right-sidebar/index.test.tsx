import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import type { HermesReadDirResult } from '@/global'
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

  // #55358: the collapse-all glyph must be the symmetric three-line one from
  // the codicon vocabulary. `collapse-all` draws two offset rounded squares —
  // at 13px its top edge reads as an asymmetric hamburger line. The header
  // action group must also stay on the shared Codicon primitive (DESIGN.md
  // iconography: no third icon set / hand-rolled SVGs inside a control group).
  it('renders the collapse-all affordance with the symmetric three-bars glyph', async () => {
    setCurrentCwd('/repo')

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    const collapse = await screen.findByRole('button', { name: 'Collapse all folders' })
    const glyph = collapse.querySelector('.codicon')
    expect(glyph).not.toBeNull()
    expect(glyph?.classList.contains('codicon-three-bars')).toBe(true)
    expect(collapse.querySelector('svg')).toBeNull()
  })
})
