import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import type { HermesReadDirResult } from '@/global'
import { $desktopActionTasks } from '@/store/activity'
import { $previewServerRestart } from '@/store/preview'
import { $connection, $sessions, $unreadFinishedSessionIds, setCurrentCwd } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

import { resetProjectTreeState } from './files/use-project-tree'

import { RightSidebarPane } from './index'

const readDir = vi.fn<(path: string) => Promise<HermesReadDirResult>>()

const session = (id: string, title: string): SessionInfo => ({
  ended_at: null,
  id,
  input_tokens: 0,
  is_active: false,
  last_active: 1_000,
  message_count: 0,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 1_000,
  title,
  tool_call_count: 0
})

function installBridge() {
  ;(window as unknown as { hermesDesktop: { readDir: typeof readDir } }).hermesDesktop = { readDir }
}

describe('RightSidebarPane', () => {
  beforeEach(() => {
    $connection.set(null)
    $desktopActionTasks.set({})
    $previewServerRestart.set(null)
    $sessionStates.set({})
    $sessions.set([])
    $unreadFinishedSessionIds.set([])
    resetProjectTreeState()
    readDir.mockReset()
    readDir.mockResolvedValue({ entries: [{ isDirectory: false, name: 'README.md', path: '/repo/README.md' }] })
    installBridge()
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    $desktopActionTasks.set({})
    $previewServerRestart.set(null)
    $sessionStates.set({})
    $sessions.set([])
    $unreadFinishedSessionIds.set([])
    setCurrentCwd('')
    resetProjectTreeState()
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('renders the tree whenever the session has a working dir (repo or not) — no picker', async () => {
    setCurrentCwd('/repo')

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} onOpenSession={vi.fn()} />)

    const refresh = await screen.findByRole('button', { name: 'Refresh tree' })

    readDir.mockClear()
    fireEvent.click(refresh)
    await waitFor(() => expect(readDir).toHaveBeenCalledWith('/repo'))

    // The freeform folder picker is retired.
    expect(screen.queryByRole('button', { name: 'Open folder' })).toBeNull()
  })

  it('shows no tree for a detached chat (no working dir)', async () => {
    setCurrentCwd('')

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} onOpenSession={vi.fn()} />)

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Refresh tree' })).toBeNull())
    expect(readDir).not.toHaveBeenCalled()
  })

  it('renders production-shaped running and finished session work across chats', () => {
    const onOpenSession = vi.fn()

    $sessions.set([session('running', 'Ship phone handoff'), session('finished', 'Audit session recovery')])
    $sessionStates.set({
      runtime: { busy: true, storedSessionId: 'running' } as ClientSessionState
    })
    $unreadFinishedSessionIds.set(['finished'])

    render(<RightSidebarPane onActivateFile={vi.fn()} onActivateFolder={vi.fn()} onOpenSession={onOpenSession} />)

    expect(screen.getByText('Ship phone handoff')).toBeTruthy()
    expect(screen.getByText('Audit session recovery')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Open Ship phone handoff' }))
    expect(onOpenSession).toHaveBeenCalledWith('running')
  })
})
