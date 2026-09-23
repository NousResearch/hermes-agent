// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { EntityIdentity, SourceHealth } from '../model'

import { EntityInspector, type EntityInspectorData, type InspectorSessionTarget } from './entity-inspector'

const IDENTITY: EntityIdentity = {
  board: 'primary',
  connectionId: 'connection-a',
  kind: 'kanban',
  profile: 'worker',
  runId: 'run-7',
  taskId: 'task-7',
  workerId: 'worker-7'
}

const SESSION: InspectorSessionTarget = {
  connectionId: 'connection-a',
  mode: 'remote',
  profile: 'worker',
  runtimeSessionId: 'runtime-1',
  sessionId: 'session-1',
  storedSessionId: 'stored-1',
  targetProfile: 'backend-worker'
}

const SOURCE: SourceHealth = {
  authority: 'partial',
  error: 'Diagnostic tail is delayed',
  observedAt: Date.UTC(2026, 7, 30, 12, 30),
  source: 'connection-a/kanban/primary'
}

function data(overrides: Partial<EntityInspectorData> = {}): EntityInspectorData {
  return {
    attachments: [{ filename: 'receipt.json', id: 'attachment-1', size: 42 }],
    blocker: { detail: 'Waiting for exact-head review', kind: 'typed-review-blocker' },
    comments: [{ author: 'operator', body: 'Keep evidence separate.', createdAt: 20, id: 'comment-1' }],
    diagnostics: [{ detail: 'The review receipt is stale.', id: 'diagnostic-1', severity: 'warning' }],
    events: [{ at: 22, id: 'event-1', kind: 'task.blocked', summary: 'Blocked after review' }],
    identity: IDENTITY,
    logTail: { content: 'pytest: 2 failed', exists: true, truncated: true },
    owningSession: SESSION,
    run: { id: 'run-7', outcome: 'blocked', status: 'failed' },
    source: SOURCE,
    subagent: {
      costUsd: 0.37,
      durationSeconds: 98,
      filesRead: ['src/a.ts'],
      filesWritten: ['src/b.ts'],
      goal: 'Review the task',
      id: 'child-7',
      state: 'failed',
      streamTail: ['Running tests', 'Found stale receipt']
    },
    task: { id: 'task-7', state: 'blocked', title: 'Repair review evidence' },
    ...overrides
  }
}

describe('EntityInspector', () => {
  it('renders exact typed identity, source, last-observed time, authority, and only available fields', () => {
    const view = render(<EntityInspector data={data({ attachments: undefined, comments: undefined })} />)

    expect(screen.getByRole('complementary', { name: 'Lunar City entity inspector' })).not.toBeNull()
    expect(screen.getByRole('region', { name: 'Identity' }).textContent).toContain('connection-a')
    expect(screen.getByRole('region', { name: 'Identity' }).textContent).toContain('task-7')
    expect(screen.getByRole('region', { name: 'Source evidence' }).textContent).toContain('connection-a/kanban/primary')
    expect(screen.getByRole('region', { name: 'Source evidence' }).textContent).toContain('Partial')
    expect(screen.getByText('Last observed').nextElementSibling?.querySelector('time')?.getAttribute('dateTime')).toBe(
      '2026-08-30T12:30:00.000Z'
    )
    expect(screen.queryByRole('region', { name: 'Attachments' })).toBeNull()
    expect(screen.queryByRole('region', { name: 'Comments' })).toBeNull()

    view.rerender(<EntityInspector data={data()} />)
    expect(screen.getByRole('region', { name: 'Attachments' })).not.toBeNull()
    expect(screen.getByRole('region', { name: 'Comments' })).not.toBeNull()
  })

  it('shows the configured title, handle, all groups, source position, and unavailable status as presentation evidence', () => {
    render(
      <EntityInspector
        data={data({
          identity: { connectionId: 'desktop-source', kind: 'profile', profile: 'test-contract-steward' },
          presentation: {
            configuredTitle: 'Test Contract Steward',
            groups: [
              { id: 'engineering', name: 'Engineering Guild' },
              { id: 'release', name: 'Acceptance & Release' }
            ],
            metadata: { observedAt: 42, source: 'profiles:desktop-source', state: 'stale' },
            placement: { lodHint: 1, overflow: true, primaryGroupId: 'engineering', slot: 27 },
            profileHandle: '@test-contract-steward',
            sourceLabel: 'Hermes Desktop'
          },
          source: { authority: 'stale', observedAt: 42, source: 'fleet:desktop-source' }
        })}
      />
    )

    const region = screen.getByRole('region', { name: 'Bot profile' })
    expect(region.textContent).toContain('Test Contract Steward')
    expect(region.textContent).toContain('@test-contract-steward')
    expect(region.textContent).toContain('Engineering Guild')
    expect(region.textContent).toContain('Acceptance & Release')
    expect(region.textContent).toContain('Hermes Desktop')
    expect(region.textContent).toContain('Unavailable')
    expect(region.textContent).toContain('Stale')
    expect(region.textContent).toContain('profiles:desktop-source')
    expect(region.textContent).toContain('Aggregate LOD')
  })

  it('keeps task, run, diagnostics, comments, events, log, attachments, subagent, files, and blocker evidence separate', () => {
    render(<EntityInspector data={data()} />)

    for (const name of [
      'Task',
      'Run',
      'Diagnostics',
      'Comments',
      'Events',
      'Log tail',
      'Attachments',
      'Subagent',
      'Files',
      'Blocker'
    ]) {
      expect(screen.getByRole('region', { name })).not.toBeNull()
    }

    expect(screen.getByRole('region', { name: 'Task' }).textContent).not.toContain('pytest: 2 failed')
    expect(screen.getByRole('region', { name: 'Diagnostics' }).textContent).not.toContain('operator')
    expect(screen.getByRole('region', { name: 'Run' }).textContent).not.toContain('typed-review-blocker')
    expect(screen.getByRole('region', { name: 'Subagent' }).textContent).toContain('$0.37')
    expect(screen.getByRole('region', { name: 'Files' }).textContent).toContain('src/b.ts')
  })

  it('opens a complete exact SessionOwnerRoute instead of a detached session ID', () => {
    const onInspectEvidence = vi.fn()
    const onOpenSession = vi.fn()

    render(<EntityInspector data={data()} onInspectEvidence={onInspectEvidence} onOpenSession={onOpenSession} />)

    fireEvent.click(
      screen.getByRole('button', {
        name: 'Open session session-1 on connection connection-a with profile worker'
      })
    )
    fireEvent.click(screen.getByRole('button', { name: 'Inspect diagnostics for task-7' }))

    expect(onOpenSession).toHaveBeenCalledWith(SESSION)
    expect(onInspectEvidence).toHaveBeenCalledWith('diagnostics', IDENTITY)
  })

  it('labels and opens only its frozen canonical owner-route copy after the input mutates', () => {
    const mutableSession = { ...SESSION }
    const onOpenSession = vi.fn()

    render(<EntityInspector data={data({ owningSession: mutableSession })} onOpenSession={onOpenSession} />)

    const button = screen.getByRole('button', {
      name: 'Open session session-1 on connection connection-a with profile worker'
    })

    mutableSession.connectionId = 'connection-z'
    mutableSession.mode = 'local'
    mutableSession.profile = 'foreign'
    mutableSession.sessionId = 'session-z'
    mutableSession.runtimeSessionId = 'runtime-z'
    mutableSession.storedSessionId = 'stored-z'
    mutableSession.targetProfile = 'foreign-backend'
    fireEvent.click(button)

    expect(button.getAttribute('aria-label')).toBe(
      'Open session session-1 on connection connection-a with profile worker'
    )
    expect(onOpenSession).toHaveBeenCalledWith(SESSION)

    const opened = onOpenSession.mock.calls[0]?.[0]

    expect(opened).not.toBe(mutableSession)
    expect(Object.isFrozen(opened)).toBe(true)
  })

  it('keeps duplicate session IDs on different connections distinct', () => {
    const onOpenSession = vi.fn()

    const secondIdentity: EntityIdentity = {
      board: 'secondary',
      connectionId: 'connection-b',
      kind: 'kanban',
      profile: 'worker',
      taskId: 'task-8'
    }

    const secondSession: InspectorSessionTarget = {
      connectionId: 'connection-b',
      profile: 'worker',
      sessionId: 'session-1'
    }

    render(
      <>
        <EntityInspector data={data()} onOpenSession={onOpenSession} />
        <EntityInspector
          data={data({ identity: secondIdentity, owningSession: secondSession })}
          onOpenSession={onOpenSession}
        />
      </>
    )

    fireEvent.click(
      screen.getByRole('button', {
        name: 'Open session session-1 on connection connection-b with profile worker'
      })
    )

    expect(onOpenSession).toHaveBeenCalledWith(secondSession)
    expect(onOpenSession).not.toHaveBeenCalledWith(SESSION)
  })

  it('fails closed when an owning session route is incomplete or mismatched', () => {
    const onOpenSession = vi.fn()

    const view = render(
      <EntityInspector
        data={data({ owningSession: { ...SESSION, connectionId: 'connection-b' } })}
        onOpenSession={onOpenSession}
      />
    )

    expect(screen.queryByRole('button', { name: /Open session/ })).toBeNull()

    view.rerender(
      <EntityInspector data={data({ owningSession: { ...SESSION, sessionId: '' } })} onOpenSession={onOpenSession} />
    )
    expect(screen.queryByRole('button', { name: /Open session/ })).toBeNull()
  })
})
