import type { JarvisEvent, JarvisTaskPhase, JarvisUiState, JarvisVoiceState } from './types'

const ACTIVITY_LIMIT = 50

type JarvisEventPatch = {
  voice?: JarvisVoiceState
  taskPhase?: JarvisTaskPhase
  activeTool?: 'event' | null
}

const EVENT_PATCHES: Record<string, JarvisEventPatch> = {
  'voice.listening': { voice: 'listening' },
  'voice.speaking': { voice: 'speaking', taskPhase: 'running' },
  'voice.stopped': { voice: 'idle' },
  'task.planning': { taskPhase: 'planning' },
  'task.running': { taskPhase: 'running' },
  'task.approval': { taskPhase: 'approval' },
  'task.cancelling': { taskPhase: 'cancelling' },
  'task.cancelled': { taskPhase: 'cancelled' },
  'task.failed': { taskPhase: 'failed' },
  'task.verified': { taskPhase: 'verified' },
  'tool.started': { activeTool: 'event' },
  'tool.completed': { activeTool: null },
}

export function initialJarvisUiState(): JarvisUiState {
  return {
    sessionId: null,
    voice: 'idle',
    task: { id: null, phase: 'idle' },
    activeTool: null,
    activity: [],
  }
}

export function reduceJarvisEvent(state: JarvisUiState, event: JarvisEvent): JarvisUiState {
  if (state.sessionId !== null && event.sessionId !== state.sessionId) {
    return state
  }

  const latestActivityAt = state.activity.at(-1)?.at

  if (latestActivityAt !== undefined && event.at < latestActivityAt) {
    return state
  }

  const patch = EVENT_PATCHES[event.type]

  if (
    state.task.id !== null &&
    event.taskId !== undefined &&
    event.taskId !== state.task.id &&
    !isTaskStartEvent(event)
  ) {
    return state
  }

  const taskId = event.taskId ?? state.task.id
  const taskPhase = patch?.taskPhase ?? state.task.phase

  const nextTask = taskId === state.task.id && taskPhase === state.task.phase
    ? state.task
    : { id: taskId, phase: taskPhase }

  const nextActiveTool = activeToolFromPatch(state, event, patch)
  const result = resultFromEvent(state, event, nextTask)
  const nextActivity = [...state.activity, event].slice(-ACTIVITY_LIMIT)

  return {
    ...state,
    sessionId: event.sessionId,
    voice: patch?.voice ?? state.voice,
    task: nextTask,
    activeTool: nextActiveTool,
    result,
    activity: nextActivity,
  }
}

function verifiedResult(event: JarvisEvent): string | undefined {
  const value = (event.detail ?? event.label ?? '').trim()

  return value || undefined
}

function isTaskStartEvent(event: JarvisEvent): boolean {
  return event.type === 'task.planning' || event.type === 'task.running'
}

function resultFromEvent(
  state: JarvisUiState,
  event: JarvisEvent,
  nextTask: JarvisUiState['task'],
): string | undefined {
  if (event.type === 'task.verified') {
    return verifiedResult(event)
  }

  if (nextTask.id !== state.task.id || event.type === 'task.planning' || event.type === 'task.running') {
    return undefined
  }

  if (event.type === 'task.failed' || event.type === 'task.cancelled' || event.type === 'task.cancelling') {
    return undefined
  }

  return state.result
}

function activeToolFromPatch(
  state: JarvisUiState,
  event: JarvisEvent,
  patch: JarvisEventPatch | undefined,
): JarvisUiState['activeTool'] {
  if (patch?.activeTool === null) {
    return null
  }

  if (patch?.activeTool === 'event' && event.toolCallId !== undefined) {
    return { id: event.toolCallId, label: event.label ?? event.toolCallId }
  }

  return state.activeTool
}

export type { JarvisEvent, JarvisUiState } from './types'
