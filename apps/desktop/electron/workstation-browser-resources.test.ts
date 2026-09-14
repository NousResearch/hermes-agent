import assert from 'node:assert/strict'

import { test } from 'vitest'

import { buildWorkstationResourceSnapshot } from './workstation-browser-resources'

const baseState = {
  runtime: 'electron-chromium' as const,
  ready: true,
  paused: false,
  controlOwner: 'agent' as const,
  controlReady: true,
  lastError: null,
  tabs: [{
    id: 'tab-1',
    title: 'Example',
    url: 'https://example.test/',
    active: false,
    loading: false,
    canGoBack: false,
    canGoForward: false,
    crashed: false,
    ownerTaskId: 'task-1'
  }],
  tasks: [{
    taskId: 'task-1',
    createdAt: '2026-09-11T12:00:00.000Z',
    updatedAt: '2026-09-11T12:01:00.000Z',
    panelHost: 'hub',
    controlHost: 'agent',
    sessionHost: 'session-1',
    kanbanCardId: 'card-1',
    runId: 'run-1',
    localConnection: 'local',
    status: 'hidden' as const,
    leaseState: null,
    parked: false,
    recoveryState: 'fresh' as const
  }]
}

test('resource projection preserves BrowserTask lineage and live evidence', () => {
  const snapshot = buildWorkstationResourceSnapshot(
    baseState,
    () => [{
      event_id: 'event-1',
      kind: 'task_started',
      task_id: 'task-1',
      session_id: 'session-1',
      message: 'started',
      timestamp: '2026-09-11T12:01:30.000Z'
    }],
    () => '2026-09-11T12:02:00.000Z'
  )

  const task = snapshot.resources.find(resource => resource.resource_type === 'browser_task')
  const journal = snapshot.resources.find(resource => resource.resource_type === 'execution_journal')

  assert.equal(snapshot.schema_version, 1)
  assert.equal(task?.resource_id, 'browser-task:task-1')
  assert.equal(task?.session_id, 'session-1')
  assert.equal(task?.state.execution_status, 'running')
  assert.deepEqual(task?.state.lineage, {
    task_id: 'task-1',
    session_id: 'session-1',
    kanban_card_id: 'card-1',
    run_id: 'run-1'
  })
  assert.deepEqual(task?.state.evidence, [
    'browser://controller',
    'browser://tab/tab-1',
    'workstation://task/task-1'
  ])
  assert.equal(journal?.state.event_count, 1)
  assert.equal(journal?.state.timeline_uri, 'workstation://journal/task-1')
})

test('resource projection fails closed when browser evidence is absent', () => {
  const snapshot = buildWorkstationResourceSnapshot(
    {
      ...baseState,
      controlReady: false,
      tabs: [],
      tasks: [{ ...baseState.tasks[0], status: 'parked', parked: true, recoveryState: 'restored' }]
    },
    () => [],
    () => '2026-09-11T12:02:00.000Z'
  )

  const task = snapshot.resources.find(resource => resource.resource_type === 'browser_task')

  assert.equal(task?.state.execution_status, 'stalled')
  assert.deepEqual(task?.state.evidence, ['workstation://task/task-1'])
  assert.deepEqual(task?.permissions, ['read'])
})

test('resource projection bounds journal details while retaining total count', () => {
  const events = Array.from({ length: 250 }, (_, index) => ({
    event_id: `event-${index}`,
    kind: 'progress',
    task_id: 'task-1',
    session_id: 'session-1',
    message: `event ${index}`,
    timestamp: `2026-09-11T12:0${String(index % 10)}:00.000Z`
  }))
  const snapshot = buildWorkstationResourceSnapshot(baseState, () => events, () => '2026-09-11T12:02:00.000Z')
  const journal = snapshot.resources.find(resource => resource.resource_type === 'execution_journal')
  const latest = journal?.state.latest_event as { event_id?: string } | null

  assert.equal(journal?.state.event_count, 250)
  assert.equal(journal?.state.visible_event_count, 200)
  assert.equal(latest?.event_id, 'event-249')
})
