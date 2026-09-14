import type { BrowserTask } from './workstation-browser-task'

import type { WorkstationBrowserState } from './workstation-browser-runtime'

export const WORKSTATION_RESOURCE_SCHEMA_VERSION = 1
export const WORKSTATION_EVENT_SCHEMA_VERSION = 1
const MAX_JOURNAL_EVENTS_IN_RESOURCE = 200

export interface WorkstationResource {
  resource_type: 'browser' | 'browser_task' | 'execution_journal'
  resource_id: string
  task_id: string | null
  session_id: string | null
  permissions: string[]
  state: Record<string, unknown>
  updated_at: string
}

export interface WorkstationResourceSnapshot {
  schema_version: typeof WORKSTATION_RESOURCE_SCHEMA_VERSION
  runtime: 'electron-chromium'
  generated_at: string
  resources: WorkstationResource[]
}

export interface WorkstationJournalEvent {
  event_id?: string
  kind?: string
  task_id?: string
  session_id?: string
  message?: string
  timestamp?: string
  elapsed_seconds?: number
  url?: string | null
  browser_tab_id?: string | null
  metadata?: Record<string, unknown>
  evidence?: Array<Record<string, unknown>>
}

export interface WorkstationEventSnapshot {
  schema_version: typeof WORKSTATION_EVENT_SCHEMA_VERSION
  runtime: 'electron-chromium'
  generated_at: string
  task_id: string | null
  events: WorkstationJournalEvent[]
}

type ResourceState = Pick<WorkstationBrowserState, 'runtime' | 'ready' | 'paused' | 'controlOwner' | 'controlReady' | 'tabs' | 'tasks' | 'lastError'>

function taskSessionId(task: BrowserTask): string | null {
  return task.sessionHost?.trim() || null
}

function taskTabId(state: ResourceState, taskId: string): string | null {
  return state.tabs.find(tab => tab.ownerTaskId === taskId)?.id ?? null
}

function taskHasLiveEvidence(state: ResourceState, taskId: string): boolean {
  const tab = state.tabs.find(candidate => candidate.ownerTaskId === taskId)

  return Boolean(state.controlReady && tab && !tab.crashed)
}

function taskExecutionStatus(state: ResourceState, task: BrowserTask): string {
  if (state.paused) {return 'hold'}
  if (task.leaseState === 'waiting' || task.sessionHost?.includes('waiting')) {
    return 'waiting-for-human'
  }
  if (!taskHasLiveEvidence(state, task.taskId)) {return 'stalled'}

  return 'running'
}

function taskResource(state: ResourceState, task: BrowserTask, updatedAt: string): WorkstationResource {
  const tabId = taskTabId(state, task.taskId)
  const live = taskHasLiveEvidence(state, task.taskId)
  const evidence = [
    state.controlReady ? 'browser://controller' : null,
    live && tabId ? `browser://tab/${tabId}` : null,
    `workstation://task/${encodeURIComponent(task.taskId)}`
  ].filter((item): item is string => Boolean(item))

  return {
    resource_type: 'browser_task',
    resource_id: `browser-task:${task.taskId}`,
    task_id: task.taskId,
    session_id: taskSessionId(task),
    permissions: state.controlOwner === 'agent' && state.controlReady && !state.paused ? ['read', 'agent-control'] : ['read'],
    state: {
      browser_status: task.status,
      execution_status: taskExecutionStatus(state, task),
      lease_state: task.leaseState,
      recovery_state: task.recoveryState,
      parked: task.parked,
      tab_id: tabId,
      kanban_card_id: task.kanbanCardId,
      run_id: task.runId,
      local_connection: task.localConnection,
      evidence,
      lineage: {
        task_id: task.taskId,
        session_id: taskSessionId(task),
        kanban_card_id: task.kanbanCardId,
        run_id: task.runId
      }
    },
    updated_at: task.updatedAt || updatedAt
  }
}

function journalResource(
  task: BrowserTask,
  events: WorkstationJournalEvent[],
  updatedAt: string
): WorkstationResource {
  const bounded = events.slice(-MAX_JOURNAL_EVENTS_IN_RESOURCE)
  const latest = bounded[bounded.length - 1] ?? null

  return {
    resource_type: 'execution_journal',
    resource_id: `execution-journal:${task.taskId}`,
    task_id: task.taskId,
    session_id: taskSessionId(task),
    permissions: ['read'],
    state: {
      event_count: events.length,
      visible_event_count: bounded.length,
      latest_event: latest,
      timeline_uri: `workstation://journal/${encodeURIComponent(task.taskId)}`
    },
    updated_at: latest?.timestamp || task.updatedAt || updatedAt
  }
}

/**
 * Build the UI-neutral projection shared by Desktop IPC and the Dashboard.
 *
 * The browser runtime and ExecutionJournal remain authoritative. This helper
 * only derives resources from their current snapshots and deliberately keeps
 * the full timeline behind the existing task-journal inspection API.
 */
export function buildWorkstationResourceSnapshot(
  state: ResourceState,
  journalReader: (taskId: string) => WorkstationJournalEvent[],
  now: () => string = () => new Date().toISOString()
): WorkstationResourceSnapshot {
  const generatedAt = now()
  const resources: WorkstationResource[] = [{
    resource_type: 'browser',
    resource_id: 'browser:electron-chromium',
    task_id: null,
    session_id: null,
    permissions: state.controlOwner === 'agent' && state.controlReady && !state.paused ? ['read', 'agent-control'] : ['read'],
    state: {
      runtime: state.runtime,
      ready: state.ready,
      paused: state.paused,
      control_owner: state.controlOwner,
      control_ready: state.controlReady,
      task_count: state.tasks.length,
      tab_count: state.tabs.length,
      last_error: state.lastError
    },
    updated_at: generatedAt
  }]

  for (const task of state.tasks) {
    resources.push(taskResource(state, task, generatedAt))
    resources.push(journalResource(task, journalReader(task.taskId), generatedAt))
  }

  return {
    schema_version: WORKSTATION_RESOURCE_SCHEMA_VERSION,
    runtime: 'electron-chromium',
    generated_at: generatedAt,
    resources
  }
}
