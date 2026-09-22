import { describe, expect, it } from 'vitest'

import type { CompletionEvent } from '../../plugins/kanban/completion-notify'

import {
  classifyTaskCondition,
  dedupeWorldEvents,
  normalizeAgentNotice,
  normalizeApprovalEvent,
  normalizeCreditEvent,
  normalizeExternalEvent,
  normalizeGatewayEvent,
  normalizeKanbanEvent,
  normalizeProfileLifecycleEvent,
  normalizePullRequestEvent,
  normalizeReleaseEvent
} from './world-events'

const now = 1_700_000_000_000

function kanban(kind: string, id = 41): CompletionEvent {
  return {
    id,
    kind,
    payload: { reason: 'dependency failed', summary: 'work finished', created_at: now / 1_000 },
    task_id: 'task-7'
  }
}

describe('world event normalization', () => {
  it.each([
    ['blocked', 'task.blocked', 'warning', 'task'],
    ['crashed', 'worker.crashed', 'error', 'worker'],
    ['gave_up', 'worker.gave_up', 'error', 'task'],
    ['timed_out', 'worker.timed_out', 'warning', 'task'],
    ['block_loop_detected', 'task.block_loop', 'critical', 'district'],
    ['completed', 'task.completed', 'success', 'task']
  ])('normalizes %s to %s', (kind, expectedKind, severity, scope) => {
    const event = normalizeKanbanEvent('main', kanban(kind), now)

    expect(event).toMatchObject({
      kind: expectedKind,
      severity,
      scope,
      sourceRef: { board: 'main', taskId: 'task-7' }
    })
    expect(event.detail).toBe('dependency failed')
  })

  it('uses an explicit fallback for unknown Kanban kinds', () => {
    const event = normalizeKanbanEvent('main', kanban('future_backend_event'), now)

    expect(event).toMatchObject({
      kind: 'system.unclassified_alert',
      severity: 'warning',
      scope: 'city',
      facts: { backendKind: 'future_backend_event' }
    })
  })

  it('normalizes notices without changing their source text', () => {
    expect(
      normalizeAgentNotice(
        { key: 'credits.depleted', level: 'error', text: 'credits paused · account action required' },
        now
      )
    ).toMatchObject({
      id: 'agent_notice:credits.depleted',
      kind: 'credits.depleted',
      severity: 'error',
      scope: 'city',
      detail: 'credits paused · account action required'
    })
  })

  it('ignores empty notices and preserves external source identity', () => {
    expect(normalizeAgentNotice({ level: 'info', text: ' ' }, now)).toBeNull()
    expect(
      normalizeExternalEvent(
        {
          id: 'pr-9',
          kind: 'pr.merged_stable',
          occurredAt: now - 100,
          scope: 'city',
          source: 'pull_request',
          title: 'Stable merge',
          sourceRef: { prId: 'pr-9' }
        },
        now
      )
    ).toMatchObject({ id: 'pull_request:pr-9', kind: 'pr.merged_stable', sourceRef: { prId: 'pr-9' } })
  })

  it('derives blocked and stale-worker conditions from current task state', () => {
    const blocked = classifyTaskCondition({ id: 'blocked', status: 'blocked', title: 'Fix auth' }, now)

    const stale = classifyTaskCondition(
      {
        assignee: 'worker-a',
        id: 'running',
        last_heartbeat_at: now / 1_000 - 121,
        status: 'running',
        title: 'Build release'
      },
      now
    )

    expect(blocked).toMatchObject({ kind: 'task.blocked', severity: 'warning', scope: 'task' })
    expect(stale).toMatchObject({ kind: 'worker.stale', severity: 'warning', scope: 'worker' })
  })

  it('deduplicates by source and id while keeping separate sources distinct', () => {
    const first = normalizeKanbanEvent('main', kanban('blocked', 41), now)
    const replacement = normalizeKanbanEvent('main', kanban('completed', 41), now + 1)

    const external = normalizeExternalEvent(
      { id: '41', kind: 'pr.approved', source: 'pull_request', title: 'Approved' },
      now + 2
    )

    expect(dedupeWorldEvents([first], [replacement, external])).toHaveLength(2)
    expect(dedupeWorldEvents([first], [replacement])).toEqual([replacement])
  })

  it('normalizes pull request and release lifecycle events into scene-ready kinds', () => {
    expect(
      normalizePullRequestEvent({ id: 'pr-9', status: 'merged', title: 'Ship stable' }, now)
    ).toMatchObject({
      kind: 'pr.merged_stable',
      scope: 'city',
      severity: 'success',
      source: 'pull_request',
      sourceRef: { prId: 'pr-9' }
    })
    expect(
      normalizePullRequestEvent({ id: 'pr-10', status: 'checks_failed', title: 'Fix tests' }, now)
    ).toMatchObject({
      kind: 'pr.review_findings',
      severity: 'error'
    })
    expect(normalizeReleaseEvent({ id: 'release-1', name: 'Desktop release', status: 'failed' }, now)).toMatchObject({
      kind: 'release.failed',
      scope: 'city',
      severity: 'critical'
    })
  })

  it('normalizes gateway, approval, credit, and profile lifecycle alerts', () => {
    expect(normalizeGatewayEvent({ id: 'gateway-1', status: 'disconnected' }, now)).toMatchObject({
      kind: 'gateway.disconnected',
      severity: 'critical'
    })
    expect(normalizeApprovalEvent({ actionId: 'a1', id: 'approval-1', status: 'required' }, now)).toMatchObject({
      kind: 'approval.required',
      severity: 'info'
    })
    expect(normalizeCreditEvent({ id: 'credits-1', remainingPercent: 0, status: 'depleted' }, now)).toMatchObject({
      kind: 'credits.depleted',
      scope: 'city',
      severity: 'critical'
    })
    expect(
      normalizeProfileLifecycleEvent({ id: 'profile-1', profile: 'reviewer', role: 'review', status: 'working' }, now)
    ).toMatchObject({
      kind: 'task.running',
      sourceRef: { agentId: 'reviewer' }
    })
  })
})
