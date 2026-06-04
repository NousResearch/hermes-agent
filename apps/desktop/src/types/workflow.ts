export type WorkflowNodeStatus =
  | 'created'
  | 'ready'
  | 'queued'
  | 'running'
  | 'reviewing'
  | 'waiting_user_confirm'
  | 'completed'
  | 'revision_needed'
  | 'failed'
  | 'retrying'
  | 'skipped'
  | 'aborted'

export type ExecutionMode = 'single_step' | 'semi_auto' | 'auto'
export type ExecutionRunStatus = 'idle' | 'running' | 'paused' | 'waiting_user_confirm' | 'completed' | 'failed' | 'stopped'

export type StreamEventType =
  | 'process_summary'
  | 'tool_call'
  | 'stage_result'
  | 'ai_reply'
  | 'node_status'
  | 'snapshot'
  | 'approval'
  | 'error'

export interface WorkflowPosition {
  x: number
  y: number
}

export interface ReviewRules {
  required: boolean
  checklist: string[]
}

export interface WorkflowNode {
  id: string
  type: string
  title: string
  description: string
  position: WorkflowPosition
  status: WorkflowNodeStatus
  inputs: Record<string, unknown>
  outputs: Record<string, unknown>
  reviewRules: ReviewRules
  skills: string[]
  model: string | null
  promptOverride: string | null
  skillMode: 'auto' | 'manual'
  references: string[]
  modelOverride: string | null
  fileChanges: NodeFileChange[]
  artifacts: string[]
  optional: boolean
  maxRetries: number
  retryCount: number
  agentSessionId: string | null
  lastRunId: string | null
  llmGenerated: boolean
}

export interface NodeFileChange {
  path: string
  status: string
  diff: string
  truncated: boolean
  isArtifact: boolean
}

export interface WorkflowEdge {
  id: string
  source: string
  target: string
  type: string
  label: string
  optional: boolean
}

export interface Workflow {
  id: string
  title: string
  nodes: WorkflowNode[]
  edges: WorkflowEdge[]
  updatedAt: number
}

export interface WorkflowProject {
  id: string
  name: string
  root: string
  goal: string
  createdAt: number
  updatedAt: number
  currentRunId: string | null
  agentSessionId: string | null
  lastOpenedAt: number | null
  archived: boolean
}

export interface ReferenceItem {
  id: string
  name: string
  path: string
  enabled: boolean
  kind: string
  addedAt: number
}

export interface SkillBinding {
  id: string
  name: string
  enabled: boolean
  source: string
}

export interface Artifact {
  id: string
  nodeId: string | null
  name: string
  path: string
  kind: string
  createdAt: number
}

export interface VersionSnapshot {
  id: string
  label: string
  reason: string
  commit: string | null
  createdAt: number
}

export interface StreamEvent {
  id: string
  projectId: string
  runId: string | null
  nodeId: string | null
  type: StreamEventType
  label: string
  timestamp: number
  summary: string
  details: Record<string, unknown>
  status: string
  durationMs: number | null
}

export interface ExecutionRun {
  id: string
  projectId: string
  mode: ExecutionMode
  status: ExecutionRunStatus
  currentNodeId: string | null
  maxConcurrency: number
  startedAt: number
  updatedAt: number
  completedAt: number | null
}

export interface ProjectBundle {
  project: WorkflowProject
  workflow: Workflow
  references: ReferenceItem[]
  skills: SkillBinding[]
  artifacts: Artifact[]
  snapshots: VersionSnapshot[]
  latestRun: ExecutionRun | null
  error: string | null
}

export interface ProjectListResponse {
  projects: WorkflowProject[]
}

export interface WorkflowEventsResponse {
  events: StreamEvent[]
}

export interface WorkflowFilesResponse {
  tree: WorkflowFileNode[]
}

export interface WorkflowFileNode {
  name: string
  path: string
  kind: 'file' | 'folder'
  children?: WorkflowFileNode[]
}

export interface WorkflowRunResponse {
  ok: boolean
  run: ExecutionRun
  project?: WorkflowProject
  workflow?: Workflow
}

export interface WorkflowComposerCompletionItem {
  type: 'file' | 'slash'
  text: string
  label: string
  detail?: string
  path?: string
}

export interface WorkflowSlashCommandItem {
  name: string
  description: string
  category: string
  argsHint?: string
  aliases: string[]
}

export interface WorkflowIntakeMessage {
  role: 'assistant' | 'user'
  content: string
  timestamp: number
}

export interface WorkflowIntakeResponse {
  ok: boolean
  intakeId: string
  messages: WorkflowIntakeMessage[]
  ready: boolean
  summary: string
}

export interface WorkflowIntakePayload {
  goal?: string
  name: string
  references?: string[]
  root?: string
}
