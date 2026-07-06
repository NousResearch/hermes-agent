import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $attentionSessionIds, $workingSessionIds } from '@/store/session'
import { $subagentsBySession, type SubagentProgress } from '@/store/subagents'
import { $todosBySession } from '@/store/todos'
import { $workstreamMetadata } from '@/store/workstream-metadata'
import type { SessionInfo } from '@/types/hermes'

import { SidebarSessionRow } from './session-row'

const session = (over: Partial<SessionInfo> = {}): SessionInfo => ({
  archived: false,
  cwd: null,
  ended_at: null,
  id: 's1',
  input_tokens: 0,
  is_active: false,
  last_active: 0,
  message_count: 1,
  model: null,
  output_tokens: 0,
  preview: null,
  source: null,
  started_at: 0,
  title: 'Desktop customizations',
  tool_call_count: 0,
  ...over
})

const noop = () => undefined

const subagent = (id: string, status: SubagentProgress['status']): SubagentProgress => ({
  filesRead: [],
  filesWritten: [],
  goal: id,
  id,
  parentId: null,
  startedAt: 0,
  status,
  stream: [],
  taskCount: 1,
  taskIndex: 0,
  updatedAt: 0
})

const renderRow = (info = session(), props: { isWorking?: boolean } = {}) =>
  render(
    <SidebarSessionRow
      isPinned={false}
      isSelected={false}
      isWorking={props.isWorking ?? false}
      onArchive={noop}
      onDelete={noop}
      onPin={noop}
      onResume={noop}
      session={info}
    />
  )

describe('SidebarSessionRow workstream badges', () => {
  beforeEach(() => {
    $attentionSessionIds.set([])
    $workingSessionIds.set([])
    $todosBySession.set({})
    $subagentsBySession.set({})
    $workstreamMetadata.set({})
  })

  afterEach(() => cleanup())

  it('renders needs-input as the dominant workstream status', () => {
    $attentionSessionIds.set(['s1'])
    $todosBySession.set({ s1: [{ id: 'todo', content: 'Do it', status: 'pending' }] })

    renderRow()

    expect(screen.queryByTitle('needs your input')).not.toBeNull()
    expect(screen.queryByText('1 todo')).not.toBeNull()
  })

  it('renders delegate status and active subagent count', () => {
    $subagentsBySession.set({ s1: [subagent('a1', 'running'), subagent('a2', 'queued')] })

    renderRow()

    expect(screen.queryByTitle('delegating to subagents')).not.toBeNull()
    expect(screen.queryByText('2 agents')).not.toBeNull()
  })

  it('keeps the existing working row indicator beside the new workstream badge', () => {
    $workingSessionIds.set(['s1'])

    const { container } = renderRow(session(), { isWorking: true })

    expect(container.querySelector('[data-working="true"]')).not.toBeNull()
    expect(container.querySelector('.arc-border')).not.toBeNull()
    expect(screen.queryByTitle('working on it now')).not.toBeNull()
  })

  it('keeps needs-input dominant and suppresses the working arc while blocked on input', () => {
    $attentionSessionIds.set(['s1'])
    $workingSessionIds.set(['s1'])

    const { container } = renderRow(session(), { isWorking: true })

    expect(container.querySelector('[data-working="true"]')).not.toBeNull()
    expect(container.querySelector('.arc-border')).toBeNull()
    expect(screen.queryByTitle('needs your input')).not.toBeNull()
  })

  it('does not render stale todo counts for completed or cancelled lingered todos', () => {
    $todosBySession.set({
      s1: [
        { id: 'done', content: 'Done', status: 'completed' },
        { id: 'cancelled', content: 'Cancelled', status: 'cancelled' }
      ]
    })

    renderRow()

    expect(screen.queryByTitle('all done')).not.toBeNull()
    expect(screen.queryByText('2 todos')).toBeNull()
    expect(screen.queryByText('1 todo')).toBeNull()
  })

  it('renders desktop lifecycle metadata as the row badge', () => {
    $workingSessionIds.set(['s1'])
    $workstreamMetadata.set({ s1: { lifecycle: 'safe_delete', updatedAt: 123 } })

    renderRow()

    expect(screen.queryByTitle('safe to delete')).not.toBeNull()
    expect(screen.queryByTitle('working on it now')).toBeNull()
  })
})
