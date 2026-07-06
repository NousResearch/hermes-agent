import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeSessionId, $attentionSessionIds, $selectedStoredSessionId, $sessions, $workingSessionIds } from '@/store/session'
import { $subagentsBySession } from '@/store/subagents'
import { $todosBySession } from '@/store/todos'
import { $workstreamMetadata } from '@/store/workstream-metadata'
import type { SessionInfo } from '@/types/hermes'

import { MissionControlView } from './index'

const session = (id: string, title = id): SessionInfo => ({
  archived: false,
  cwd: null,
  ended_at: null,
  id,
  input_tokens: 0,
  is_active: false,
  last_active: 0,
  message_count: 1,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 0,
  title,
  tool_call_count: 0
})

beforeEach(() => {
  $activeSessionId.set(null)
  $attentionSessionIds.set([])
  $selectedStoredSessionId.set(null)
  $sessions.set([])
  $subagentsBySession.set({})
  $todosBySession.set({})
  $workingSessionIds.set([])
  $workstreamMetadata.set({})
})

afterEach(cleanup)

describe('MissionControlView', () => {
  it('renders the empty cockpit', () => {
    render(<MissionControlView onClose={vi.fn()} onOpenSession={vi.fn()} />)

    expect(screen.getByText('Mission Control')).toBeTruthy()
    expect(screen.getByText('No workstreams')).toBeTruthy()
  })

  it('keeps closed and safe-delete sessions in separate cards', () => {
    $sessions.set([session('closed', 'Closed topic'), session('safe', 'Safe delete topic')])
    $workstreamMetadata.set({
      closed: { lifecycle: 'closed', updatedAt: 1 },
      safe: { lifecycle: 'safe_delete', updatedAt: 1 }
    })

    render(<MissionControlView onClose={vi.fn()} onOpenSession={vi.fn()} />)

    expect(screen.getByText('Closed')).toBeTruthy()
    expect(screen.getByText('Safe delete')).toBeTruthy()
    expect(screen.getByText('Closed topic')).toBeTruthy()
    expect(screen.getByText('Safe delete topic')).toBeTruthy()
  })

  it('opens the selected session from a row click', () => {
    const openSession = vi.fn()
    $sessions.set([session('s1', 'Active workstream')])
    $todosBySession.set({ s1: [{ content: 'Do it', id: 'todo-1', status: 'pending' }] })

    render(<MissionControlView onClose={vi.fn()} onOpenSession={openSession} />)
    fireEvent.click(screen.getByText('Active workstream'))

    expect(openSession).toHaveBeenCalledWith('s1')
  })
})
