import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type { CronJob } from '@/hermes'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $clarifyRequests } from '@/store/clarify'
import { beginCronJobsRequest, commitCronJobsRequest, setCronJobs } from '@/store/cron'
import { $liveWorkRoster, setLiveWorkRosterSource } from '@/store/live-work-roster'
import { clearAllPrompts, clearApprovalRequest, setApprovalRequest } from '@/store/prompts'
import {
  $cronSessions,
  $messagingSessions,
  $sessions,
  $unlistedSessionOwnerRows,
  $unreadFinishedSessionIds
} from '@/store/session'
import { $sessionStates, clearAllSessionStates, recordSessionEventScope } from '@/store/session-states'
import { $unreadFinishedMarkers } from '@/store/session-unread'
import { $unreadWriteGuard } from '@/store/session-unread-remote'
import { makeSessionInfo } from '@/test/session-info'

import { useAgentSidebarAttention } from './use-agent-sidebar-attention'
import type { SimplifiedAgent } from './use-simplified-agents'

vi.mock('./use-simplified-agents', () => ({ useSimplifiedOwner: () => 'gateway-a' }))

const agents: SimplifiedAgent[] = ['gateway-a', 'gateway-b'].map(connectionId => ({
  id: connectionId,
  connectionId,
  profile: 'Display name',
  targetProfile: 'research',
  label: 'Same display name',
  gateway: connectionId
}))

beforeEach(() => {
  clearAllSessionStates()
  clearAllPrompts()
  $clarifyRequests.set({})
  $sessions.set([])
  $cronSessions.set([])
  $messagingSessions.set([])
  $unlistedSessionOwnerRows.set([])
  $unreadFinishedSessionIds.set([])
  $unreadFinishedMarkers.set({})
  $unreadWriteGuard.set(new Map())
  $liveWorkRoster.set({})
  setCronJobs([])
})

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  clearAllPrompts()
  $clarifyRequests.set({})
})

it('uses sibling chats and clears pending input without borrowing a same-name gateway', () => {
  const sibling = makeSessionInfo({ id: 'sibling', profile: 'research', connection_id: 'gateway-b', unread: true })
  $sessions.set([sibling])
  recordSessionEventScope({ session_id: 'runtime', connectionId: 'gateway-b', profile: 'research' })
  $sessionStates.set({ runtime: { ...createClientSessionState('sibling'), busy: true, turnStartedAt: 10_000 } })
  const { result } = renderHook(() => useAgentSidebarAttention(agents, [sibling]))
  expect(result.current.get('gateway-b')).toMatchObject({ kind: 'running', since: 10_000, session: sibling })
  expect(result.current.has('gateway-a')).toBe(false)

  act(() =>
    $clarifyRequests.set({ runtime: { sessionId: 'runtime', requestId: 'question', questions: [], receivedAt: 12.5 } })
  )
  expect(result.current.get('gateway-b')).toMatchObject({ kind: 'input', since: 12_500 })
  act(() =>
    setApprovalRequest({ sessionId: 'runtime', requestId: 'approval', command: 'deploy', description: 'Deploy' })
  )
  expect(result.current.get('gateway-b')?.kind).toBe('approval')
  expect(result.current.get('gateway-b')?.since).toBeUndefined()
  act(() => clearApprovalRequest('runtime', 'approval'))
  expect(result.current.get('gateway-b')).toMatchObject({ kind: 'input', since: 12_500 })
  act(() => $clarifyRequests.set({}))
  expect(result.current.get('gateway-b')?.kind).toBe('running')
  act(() => $sessionStates.set({ runtime: createClientSessionState('sibling') }))
  expect(result.current.get('gateway-b')?.kind).toBe('unread')
  expect(result.current.has('gateway-a')).toBe(false)
})

it('rejects ambiguous bare ids, but retains an event-proven prompt owner and source session route', () => {
  const rows = agents.map(agent =>
    makeSessionInfo({ id: 'same-id', profile: 'research', connection_id: agent.connectionId! })
  )

  $sessions.set(rows)
  $sessionStates.set({ runtime: { ...createClientSessionState('same-id'), busy: true } })
  act(() =>
    setApprovalRequest({ sessionId: 'runtime', requestId: 'approval', command: 'deploy', description: 'Deploy' })
  )
  const { result } = renderHook(() => useAgentSidebarAttention(agents, rows))
  expect(result.current.size).toBe(0)

  act(() => {
    recordSessionEventScope({ session_id: 'runtime', connectionId: 'gateway-b', profile: 'research' })
    setApprovalRequest({ sessionId: 'runtime', requestId: 'owned-approval', command: 'deploy', description: 'Deploy' })
  })
  expect(result.current.get('gateway-b')).toMatchObject({ kind: 'approval', session: { connection_id: 'gateway-b' } })
  expect(result.current.has('gateway-a')).toBe(false)
})

it('includes schedule runs, scoped schedule failures and offscreen background work without new requests', () => {
  const cron = makeSessionInfo({
    id: 'run',
    source: 'cron',
    profile: 'research',
    connection_id: 'gateway-b',
    scheduler_owned: true
  })

  const sibling = makeSessionInfo({ id: 'offscreen', profile: 'research', connection_id: 'gateway-a' })
  $cronSessions.set([cron])
  $sessions.set([sibling])
  setLiveWorkRosterSource('primary:research', [{ session_id: 'offscreen', background: 1, subagents: 0 }])
  const { result } = renderHook(() => useAgentSidebarAttention(agents, [sibling]))
  expect(result.current.get('gateway-a')?.kind).toBe('running')
  expect(result.current.get('gateway-b')).toMatchObject({ kind: 'running', session: cron })

  act(() =>
    commitCronJobsRequest(beginCronJobsRequest('gateway-b\u0000research'), [
      {
        id: 'schedule',
        enabled: true,
        last_run_at: '2026-11-10T09:00:00Z',
        latest_execution: { id: 'attempt', status: 'failed' }
      }
    ])
  )
  expect(result.current.get('gateway-b')).toEqual({ kind: 'failed', since: Date.parse('2026-11-10T09:00:00Z') })
  expect(result.current.get('gateway-a')?.kind).toBe('running')
  act(() =>
    commitCronJobsRequest(beginCronJobsRequest('gateway-b\u0000research'), [
      {
        id: 'schedule',
        enabled: true,
        last_error: 'old error',
        latest_execution: { id: 'next-attempt', status: 'success' }
      }
    ])
  )
  expect(result.current.get('gateway-b')?.kind).toBe('running')
  act(() => {
    $cronSessions.set([])
    $liveWorkRoster.set({})
  })
  expect(result.current.size).toBe(0)
})

it('attributes aggregate schedules to each profile without crossing explicit machine scopes', () => {
  const roster = [
    ...agents,
    { ...agents[1], id: 'other', targetProfile: 'other' },
    { ...agents[0], id: 'local', connectionId: null }
  ]

  const running: CronJob = {
    id: 'research-run',
    enabled: true,
    profile: 'research',
    latest_execution: { id: 'attempt', status: 'running', claimed_at: '2026-11-10T09:00:00Z' }
  }

  const failed: CronJob = {
    id: 'other-run',
    enabled: true,
    profile: 'other',
    latest_execution: { id: 'failed-attempt', status: 'failed' }
  }

  const { result } = renderHook(() => useAgentSidebarAttention(roster, []))

  act(() => commitCronJobsRequest(beginCronJobsRequest('gateway-b\u0000all'), [running, failed]))
  expect([...result.current.keys()].sort()).toEqual(['gateway-b', 'other'])
  expect(result.current.get('gateway-b')).toEqual({ kind: 'running', since: Date.parse('2026-11-10T09:00:00Z') })
  expect(result.current.get('other')).toEqual({ kind: 'failed' })

  // A concrete request remains authoritative even if a row's profile is absent or different.
  act(() => commitCronJobsRequest(beginCronJobsRequest('gateway-b\u0000research'), [failed]))
  expect([...result.current.entries()]).toEqual([['gateway-b', { kind: 'failed' }]])

  // An explicit local snapshot must not borrow the currently viewed remote connection.
  act(() => commitCronJobsRequest(beginCronJobsRequest('\u0000all'), [running]))
  expect([...result.current.keys()]).toEqual(['local'])
  act(() => commitCronJobsRequest(beginCronJobsRequest('gateway-b\u0000all'), [{ ...failed, profile: undefined }]))
  expect(result.current.size).toBe(0)
  act(() => setCronJobs([failed]))
  expect(result.current.size).toBe(0)
})

it('alerts on execution or delivery failure, but lets the current run supersede stale failures', () => {
  const { result } = renderHook(() => useAgentSidebarAttention(agents, []))
  const job: CronJob = { id: 'schedule', enabled: true, profile: 'research' }
  const completed = { id: 'attempt', status: 'completed' }

  const failures: Partial<CronJob>[] = [
    { latest_execution: { id: 'attempt', status: 'failed' } },
    { latest_execution: { ...completed, delivery_outcome: 'failed' } },
    { latest_execution: completed, last_status: 'delivery_failed' },
    { latest_execution: completed, last_delivery_error: 'Delivery refused' },
    { last_status: 'error' },
    { last_error: 'Execution failed' },
    { last_delivery_error: 'Delivery refused' }
  ]

  for (const failure of failures) {
    act(() => commitCronJobsRequest(beginCronJobsRequest('gateway-b\u0000all'), [{ ...job, ...failure }]))
    expect([...result.current.entries()]).toEqual([['gateway-b', { kind: 'failed' }]])
  }

  const stale = {
    last_status: 'delivery_failed',
    last_error: 'Previous failure',
    last_delivery_error: 'Previous delivery'
  }

  const running = { id: 'current', status: 'running', claimed_at: '2026-11-10T10:00:00Z' }

  for (const execution of [
    { latest_execution: running },
    { latest_execution: { id: 'refused', status: 'failed' }, live_execution: running }
  ]) {
    act(() => commitCronJobsRequest(beginCronJobsRequest('gateway-b\u0000all'), [{ ...job, ...stale, ...execution }]))
    expect([...result.current.entries()]).toEqual([
      ['gateway-b', { kind: 'running', since: Date.parse(running.claimed_at) }]
    ])
  }

  act(() =>
    commitCronJobsRequest(beginCronJobsRequest('gateway-b\u0000all'), [
      { ...job, ...stale, latest_execution: { ...completed, delivery_outcome: 'delivered' } }
    ])
  )
  expect(result.current.size).toBe(0)
})

it('does not assign persisted profile-only unread markers across two gateways', () => {
  $unreadFinishedMarkers.set({ research: ['unloaded'] })

  const { result, rerender } = renderHook(({ roster }) => useAgentSidebarAttention(roster, []), {
    initialProps: { roster: agents }
  })

  expect(result.current.size).toBe(0)
  rerender({ roster: [agents[1]] })
  expect(result.current.get('gateway-b')?.kind).toBe('unread')
})

it('reports only the current failed turn and clears it when a new turn begins', () => {
  const row = makeSessionInfo({ id: 'chat', profile: 'research', connection_id: 'gateway-b' })

  const failed = {
    ...createClientSessionState('chat'),
    messages: [
      { id: 'error', role: 'assistant' as const, parts: [], error: 'Provider rejected request', completedAt: 25 }
    ]
  }

  recordSessionEventScope({ session_id: 'runtime', connectionId: 'gateway-b', profile: 'research' })
  $sessionStates.set({ runtime: failed })
  const { result } = renderHook(() => useAgentSidebarAttention(agents, [row]))
  expect(result.current.get('gateway-b')).toMatchObject({ kind: 'failed', since: 25_000, session: row })
  expect(result.current.has('gateway-a')).toBe(false)
  act(() => $sessionStates.set({ runtime: { ...failed, busy: true, turnLive: true, turnStartedAt: 30_000 } }))
  expect(result.current.get('gateway-b')?.kind).toBe('running')
  act(() =>
    $sessionStates.set({
      runtime: {
        ...createClientSessionState('chat'),
        messages: [...failed.messages, { id: 'new-user', role: 'user', parts: [] }]
      }
    })
  )
  expect(result.current.size).toBe(0)
})
