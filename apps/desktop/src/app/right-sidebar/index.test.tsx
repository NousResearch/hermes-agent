import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactElement } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'
import {
  $connection,
  releaseWorkspaceCwdOwner,
  setCurrentCwd,
  setFreshDraftReady,
  setSelectedStoredSessionId,
  setWorkspaceCwdOwner
} from '@/store/session'

import { resetProjectTreeState } from './files/use-project-tree'

import { RightSidebarPane } from './index'

const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()

function installBridge() {
  ;(window as unknown as { hermesDesktop: { readDir: typeof readDir } }).hermesDesktop = { readDir }
}

function renderAt(element: ReactElement, path = '/stored-session') {
  return render(<MemoryRouter initialEntries={[path]}>{element}</MemoryRouter>)
}

describe('RightSidebarPane', () => {
  beforeEach(() => {
    $connection.set(null)
    setSelectedStoredSessionId(null)
    setWorkspaceCwdOwner(null)
    setFreshDraftReady(false)
    resetProjectTreeState()
    readDir.mockReset()
    readDir.mockResolvedValue({ entries: [{ isDirectory: false, name: 'README.md', path: '/repo/README.md' }] })
    installBridge()
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    setCurrentCwd('')
    setFreshDraftReady(false)
    setSelectedStoredSessionId(null)
    setWorkspaceCwdOwner(null)
    resetProjectTreeState()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('renders the tree whenever the session has a working dir (repo or not) — no picker', async () => {
    setCurrentCwd('/repo')

    renderAt(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    const refresh = await screen.findByRole('button', { name: 'Refresh tree' })

    readDir.mockClear()
    fireEvent.click(refresh)
    await waitFor(() => expect(readDir).toHaveBeenCalledWith('/repo'))

    // The freeform folder picker is retired.
    expect(screen.queryByRole('button', { name: 'Open folder' })).toBeNull()
  })

  it('does not read a retained cwd while it belongs to a previous session', async () => {
    setSelectedStoredSessionId('new-session')
    setWorkspaceCwdOwner('previous-session')
    setCurrentCwd('/home/doug/default-profile-workspace')

    renderAt(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Refresh tree' })).toBeNull())
    expect(readDir).not.toHaveBeenCalled()
  })

  it('shows no tree for a detached chat (no working dir)', async () => {
    setCurrentCwd('')

    renderAt(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Refresh tree' })).toBeNull())
    expect(readDir).not.toHaveBeenCalled()
  })

  it('hides the previous conversation tree while a fresh draft has no workspace owner', async () => {
    setSelectedStoredSessionId('stored-previous')
    setCurrentCwd('/repo/previous-worktree')
    setWorkspaceCwdOwner('stored-previous')
    setSelectedStoredSessionId(null)
    releaseWorkspaceCwdOwner()
    // A late runtime publication can still name the null-id draft as owner;
    // an implicit draft target must remain hidden even through that race.
    setWorkspaceCwdOwner(null)
    setFreshDraftReady(true)

    renderAt(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />, '/')

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Refresh tree' })).toBeNull())
    expect(readDir).not.toHaveBeenCalled()
  })
})
