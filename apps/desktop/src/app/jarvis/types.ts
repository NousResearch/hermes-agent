export type JarvisVoiceState = 'idle' | 'listening' | 'speaking' | 'error'
export type JarvisTaskPhase =
  | 'idle'
  | 'planning'
  | 'running'
  | 'approval'
  | 'cancelling'
  | 'cancelled'
  | 'failed'
  | 'verified'

export interface JarvisStreamEvent {
  type: string
  data?: unknown
  session_id?: string
  task_id?: string
}

export interface JarvisEvent {
  type: string
  sessionId: string
  taskId?: string
  toolCallId?: string
  at: number
  label?: string
  detail?: string
}

export interface JarvisUiState {
  sessionId: string | null
  connected: boolean
  voice: JarvisVoiceState
  task: { id: string | null; phase: JarvisTaskPhase }
  activeTool: { id: string; label: string } | null
  activity: readonly JarvisEvent[]
}
