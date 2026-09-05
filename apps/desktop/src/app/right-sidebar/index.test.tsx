import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesReadDirResult } from '@/global'
import {
  $connection,
  $selectedStoredSessionId,
  $sessions,
  $workspaceCwdOwner,
  setCurrentCwd
} from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { resetProjectTreeState } from './files/use-project-tree'

import { RightSidebarPane } from './index'

const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()

function listedSession(overrides: Partial<SessionInfo> = {}): SessionInfo {
  return {
    ended_at: null,
    id: 'new-session',
    input_tokens: 0,
    is_active: true,
    last_active: 1,
    message_count: 1,
    model: null,
    output_tokens: 0,
    preview: null,
    profile: 'default',
    source: 'desktop',
    started_at: 1,
    title: 'Ops chat',
    tool_call_count: 0,
    ...overrides
  }
}

function installBridge() {
  ;(window as unknown as { hermesDesktop: { readDir: typeof readDir } }).hermesDesktop = { readDir }
}

describe('RightSidebarPane', () => {
  beforeEach(() => {
    $connection.set(null)
    $selectedStoredSessionId.set(null)
    $workspaceCwdOwner.set(null)
    $sessions.set([])
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
    $sessions.set([])
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

  it('reads the selected session listed cwd when the ownership lock is stale', async () => {
    $selectedStoredSessionId.set('new-session')
    $workspaceCwdOwner.set('previous-session')
    setCurrentCwd('/home/doug/default-profile-workspace')
    $sessions.set([listedSession({ cwd: '/repo/ops' })])

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} />)

    const refresh = await screen.findByRole('button', { name: 'Refresh tree' })
    readDir.mockClear()
    fireEvent.click(refresh)
    await waitFor(() => expect(readDir).toHaveBeenCalledWith('/repo/ops'))
  })
})
