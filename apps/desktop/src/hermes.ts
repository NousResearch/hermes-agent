import { JsonRpcGatewayClient } from '@hermes/shared'

import type {
  ActionResponse,
  ActionStatusResponse,
  AnalyticsResponse,
  AudioSpeakResponse,
  AudioTranscriptionResponse,
  AuxiliaryModelsResponse,
  ConfigSchemaResponse,
  CronJob,
  CronJobCreatePayload,
  CronJobUpdates,
  ElevenLabsVoicesResponse,
  EnvVarInfo,
  HermesConfig,
  HermesConfigRecord,
  LogsResponse,
  MessagingPlatformsResponse,
  MessagingPlatformTestResponse,
  MessagingPlatformUpdate,
  ModelAssignmentRequest,
  ModelAssignmentResponse,
  ModelInfoResponse,
  ModelOptionsResponse,
  OAuthPollResponse,
  OAuthProvidersResponse,
  OAuthStartResponse,
  OAuthSubmitResponse,
  PaginatedSessions,
  ProfileCreatePayload,
  ProfileSetupCommand,
  ProfileSoul,
  ProfilesResponse,
  SessionMessagesResponse,
  SessionInfo,
  SessionSearchResponse,
  SkillInfo,
  StatusResponse,
  ToolsetConfig,
  ToolsetInfo
} from '@/types/hermes'
import type {
  ExecutionMode,
  ProjectBundle,
  ProjectListResponse,
  ReferenceItem,
  SkillBinding,
  Workflow,
  WorkflowComposerCompletionItem,
  WorkflowEventsResponse,
  WorkflowFilesResponse,
  WorkflowIntakeAnswer,
  WorkflowIntakePayload,
  WorkflowIntakeResponse,
  WorkflowProject,
  WorkflowSlashCommandItem,
  WorkflowRunResponse
} from '@/types/workflow'

const DEFAULT_GATEWAY_REQUEST_TIMEOUT_MS = 30_000
const WORKFLOW_LLM_REQUEST_TIMEOUT_MS = 180_000
const WORKFLOW_SESSION_PREFIXES = ['workflow-project-', 'workflow-node-'] as const

function isWorkflowSessionId(id: null | string | undefined): boolean {
  return WORKFLOW_SESSION_PREFIXES.some(prefix => String(id ?? '').startsWith(prefix))
}

function isWorkflowSessionLike(session: Pick<SessionInfo, 'id' | 'source'> & { _lineage_root_id?: null | string }): boolean {
  return session.source === 'workflow' || isWorkflowSessionId(session.id) || isWorkflowSessionId(session._lineage_root_id)
}

function isMissingOrMethodError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? '')

  return /\b(404|405)\b/.test(message) || /method not allowed|not found/i.test(message)
}

export type {
  ActionResponse,
  ActionStatusResponse,
  AnalyticsDailyEntry,
  AnalyticsModelEntry,
  AnalyticsResponse,
  AnalyticsSkillEntry,
  AnalyticsSkillsSummary,
  AnalyticsTotals,
  AudioSpeakResponse,
  AudioTranscriptionResponse,
  AuxiliaryModelsResponse,
  ConfigFieldSchema,
  ConfigSchemaResponse,
  CronJob,
  CronJobCreatePayload,
  CronJobSchedule,
  CronJobUpdates,
  ElevenLabsVoice,
  ElevenLabsVoicesResponse,
  EnvVarInfo,
  GatewayReadyPayload,
  HermesConfig,
  HermesConfigRecord,
  LogsResponse,
  MessagingEnvVarInfo,
  MessagingHomeChannel,
  MessagingPlatformInfo,
  MessagingPlatformsResponse,
  MessagingPlatformTestResponse,
  MessagingPlatformUpdate,
  ModelAssignmentRequest,
  ModelAssignmentResponse,
  ModelInfoResponse,
  ModelOptionProvider,
  ModelOptionsResponse,
  PaginatedSessions,
  ProfileCreatePayload,
  ProfileInfo,
  ProfileSetupCommand,
  ProfileSoul,
  ProfilesResponse,
  RpcEvent,
  SessionCreateResponse,
  SessionInfo,
  SessionMessage,
  SessionMessagesResponse,
  SessionResumeResponse,
  SessionRuntimeInfo,
  SessionSearchResponse,
  SessionSearchResult,
  SkillInfo,
  StatusResponse,
  ToolsetConfig,
  ToolsetInfo
} from '@/types/hermes'

export class HermesGateway extends JsonRpcGatewayClient {
  constructor() {
    super({
      closedErrorMessage: 'Hermes gateway connection closed',
      connectErrorMessage: 'Could not connect to Hermes gateway',
      createRequestId: nextId => nextId,
      notConnectedErrorMessage: 'Hermes gateway is not connected',
      requestTimeoutMs: DEFAULT_GATEWAY_REQUEST_TIMEOUT_MS
    })
  }
}

export async function listSessions(
  limit = 40,
  minMessages = 0,
  archived: 'exclude' | 'include' | 'only' = 'exclude',
  order: 'created' | 'recent' = 'recent'
): Promise<PaginatedSessions> {
  const result = await window.hermesDesktop.api<PaginatedSessions>({
    path: `/api/sessions?limit=${limit}&offset=0&min_messages=${Math.max(0, minMessages)}&archived=${archived}&order=${order}`
  })

  const sessions = result.sessions.filter(session => !isWorkflowSessionLike(session)).slice(0, limit)

  return {
    ...result,
    sessions,
    offset: 0
  }
}

export function setSessionArchived(id: string, archived: boolean): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    path: `/api/sessions/${encodeURIComponent(id)}`,
    method: 'PATCH',
    body: { archived }
  })
}

export function searchSessions(query: string): Promise<SessionSearchResponse> {
  return window.hermesDesktop
    .api<SessionSearchResponse>({
      path: `/api/sessions/search?q=${encodeURIComponent(query)}`
    })
    .then(result => ({
      ...result,
      results: result.results.filter(item => item.source !== 'workflow' && !isWorkflowSessionId(item.session_id))
    }))
}

export function getSessionMessages(id: string): Promise<SessionMessagesResponse> {
  return window.hermesDesktop.api<SessionMessagesResponse>({
    path: `/api/sessions/${encodeURIComponent(id)}/messages`
  })
}

export function deleteSession(id: string): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    path: `/api/sessions/${encodeURIComponent(id)}`,
    method: 'DELETE'
  })
}

export function renameSession(id: string, title: string): Promise<{ ok: boolean; title: string }> {
  return window.hermesDesktop.api<{ ok: boolean; title: string }>({
    path: `/api/sessions/${encodeURIComponent(id)}`,
    method: 'PATCH',
    body: { title }
  })
}

export function getGlobalModelInfo(): Promise<ModelInfoResponse> {
  return window.hermesDesktop.api<ModelInfoResponse>({
    path: '/api/model/info'
  })
}

export function getStatus(): Promise<StatusResponse> {
  return window.hermesDesktop.api<StatusResponse>({
    path: '/api/status'
  })
}

export function getLogs(params: {
  component?: string
  file?: string
  level?: string
  lines?: number
}): Promise<LogsResponse> {
  const query = new URLSearchParams()

  if (params.file) {
    query.set('file', params.file)
  }

  if (typeof params.lines === 'number') {
    query.set('lines', String(params.lines))
  }

  if (params.level && params.level !== 'ALL') {
    query.set('level', params.level)
  }

  if (params.component && params.component !== 'all') {
    query.set('component', params.component)
  }

  const suffix = query.toString()

  return window.hermesDesktop.api<LogsResponse>({
    path: suffix ? `/api/logs?${suffix}` : '/api/logs'
  })
}

export function getHermesConfig(): Promise<HermesConfig> {
  return window.hermesDesktop.api<HermesConfig>({
    path: '/api/config'
  })
}

export function getHermesConfigRecord(): Promise<HermesConfigRecord> {
  return window.hermesDesktop.api<HermesConfigRecord>({
    path: '/api/config'
  })
}

export function getHermesConfigDefaults(): Promise<HermesConfigRecord> {
  return window.hermesDesktop.api<HermesConfigRecord>({
    path: '/api/config/defaults'
  })
}

export function getHermesConfigSchema(): Promise<ConfigSchemaResponse> {
  return window.hermesDesktop.api<ConfigSchemaResponse>({
    path: '/api/config/schema'
  })
}

export function saveHermesConfig(config: HermesConfigRecord): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    path: '/api/config',
    method: 'PUT',
    body: { config }
  })
}

export function getEnvVars(): Promise<Record<string, EnvVarInfo>> {
  return window.hermesDesktop.api<Record<string, EnvVarInfo>>({
    path: '/api/env'
  })
}

export function setEnvVar(key: string, value: string): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    path: '/api/env',
    method: 'PUT',
    body: { key, value }
  })
}

export function validateProviderCredential(
  key: string,
  value: string
): Promise<{ ok: boolean; reachable: boolean; message: string }> {
  return window.hermesDesktop.api<{ ok: boolean; reachable: boolean; message: string }>({
    path: '/api/providers/validate',
    method: 'POST',
    body: { key, value }
  })
}

export function listWorkflowProjects(options: { includeArchived?: boolean } = {}): Promise<ProjectListResponse> {
  const suffix = options.includeArchived ? '?include_archived=true' : ''

  if (!window.hermesDesktop?.api) {
    return Promise.resolve({ projects: [] })
  }

  return window.hermesDesktop.api<ProjectListResponse>({
    path: `/api/workflows/projects${suffix}`
  })
}

export function createWorkflowProject(payload: {
  goal?: string
  name: string
  references?: string[]
  root?: string
}): Promise<ProjectBundle> {
  return window.hermesDesktop.api<ProjectBundle>({
    path: '/api/workflows/projects',
    method: 'POST',
    body: payload,
    timeoutMs: WORKFLOW_LLM_REQUEST_TIMEOUT_MS
  })
}

export function startWorkflowIntake(payload: WorkflowIntakePayload): Promise<WorkflowIntakeResponse> {
  return window.hermesDesktop.api<WorkflowIntakeResponse>({
    path: '/api/workflows/intake/start',
    method: 'POST',
    body: payload,
    timeoutMs: WORKFLOW_LLM_REQUEST_TIMEOUT_MS
  })
}

export function sendWorkflowIntakeMessage(intakeId: string, message: string): Promise<WorkflowIntakeResponse> {
  return window.hermesDesktop.api<WorkflowIntakeResponse>({
    path: `/api/workflows/intake/${encodeURIComponent(intakeId)}/message`,
    method: 'POST',
    body: { message },
    timeoutMs: WORKFLOW_LLM_REQUEST_TIMEOUT_MS
  })
}

export function submitWorkflowIntakeAnswers(intakeId: string, answers: WorkflowIntakeAnswer[]): Promise<WorkflowIntakeResponse> {
  return window.hermesDesktop.api<WorkflowIntakeResponse>({
    path: `/api/workflows/intake/${encodeURIComponent(intakeId)}/answers`,
    method: 'POST',
    body: { answers },
    timeoutMs: WORKFLOW_LLM_REQUEST_TIMEOUT_MS
  })
}

export function confirmWorkflowIntake(
  intakeId: string,
  payload: WorkflowIntakePayload & { projectId?: string | null; summary?: string }
): Promise<ProjectBundle> {
  return window.hermesDesktop.api<ProjectBundle>({
    path: `/api/workflows/intake/${encodeURIComponent(intakeId)}/confirm`,
    method: 'POST',
    body: { ...payload, intakeId },
    timeoutMs: WORKFLOW_LLM_REQUEST_TIMEOUT_MS
  })
}

export function openWorkflowProject(root: string): Promise<ProjectBundle> {
  return window.hermesDesktop.api<ProjectBundle>({
    path: '/api/workflows/projects/open',
    method: 'POST',
    body: { root }
  })
}

export function getWorkflowProject(projectId: string): Promise<ProjectBundle> {
  return window.hermesDesktop.api<ProjectBundle>({
    path: `/api/workflows/projects/${encodeURIComponent(projectId)}`
  })
}

export function updateWorkflowProject(
  projectId: string,
  patch: { archived?: boolean; name?: string }
): Promise<ProjectBundle> {
  return window.hermesDesktop.api<ProjectBundle>({
    path: `/api/workflows/projects/${encodeURIComponent(projectId)}`,
    method: 'PATCH',
    body: patch
  })
}

export function removeWorkflowProjectFromHistory(
  projectId: string
): Promise<{ ok: boolean; projectId: string; root: string; rootPreserved: boolean }> {
  const encoded = encodeURIComponent(projectId)

  return window.hermesDesktop
    .api<{ ok: boolean; projectId: string; root: string; rootPreserved: boolean }>({
      path: `/api/workflows/projects/${encoded}/remove-from-history`,
      method: 'POST'
    })
    .catch(error => {
      if (!isMissingOrMethodError(error)) {
        throw error
      }

      return window.hermesDesktop
        .api<{ ok: boolean; projectId: string; root: string; rootPreserved: boolean }>({
          path: `/api/workflows/projects/${encoded}`,
          method: 'DELETE'
        })
        .catch(deleteError => {
          if (!isMissingOrMethodError(deleteError)) {
            throw deleteError
          }

          return window.hermesDesktop
            .api<ProjectBundle>({
              path: `/api/workflows/projects/${encoded}`,
              method: 'PATCH',
              body: { archived: true }
            })
            .then(bundle => ({
              ok: true,
              projectId,
              root: bundle.project.root,
              rootPreserved: true
            }))
        })
    })
}

export function exportWorkflowProject(project: WorkflowProject): Promise<{ canceled: boolean; path: null | string }> {
  const safeName = (project.name || 'workflow-project').trim().replace(/[^a-z0-9._-]+/gi, '-').replace(/^-+|-+$/g, '')

  return window.hermesDesktop.downloadApiFile({
    path: `/api/workflows/projects/${encodeURIComponent(project.id)}/export`,
    defaultFilename: `${safeName || 'workflow-project'}-${project.id.slice(0, 8)}.zip`,
    dialogTitle: 'Export workflow project',
    filters: [{ name: 'Zip archives', extensions: ['zip'] }],
    timeoutMs: 120_000
  })
}

export function generateWorkflow(projectId: string): Promise<ProjectBundle> {
  return window.hermesDesktop.api<ProjectBundle>({
    path: `/api/workflows/projects/${encodeURIComponent(projectId)}/generate`,
    method: 'POST',
    timeoutMs: WORKFLOW_LLM_REQUEST_TIMEOUT_MS
  })
}

export function saveWorkflow(projectId: string, workflow: Workflow, snapshotLabel = 'Workflow edited'): Promise<ProjectBundle> {
  return window.hermesDesktop.api<ProjectBundle>({
    path: `/api/workflows/projects/${encodeURIComponent(projectId)}/workflow`,
    method: 'PUT',
    body: { workflow, snapshotLabel }
  })
}

export function startWorkflowRun(
  projectId: string,
  payload: { maxConcurrency?: number; mode: ExecutionMode }
): Promise<WorkflowRunResponse> {
  return window.hermesDesktop.api<WorkflowRunResponse>({
    path: `/api/workflows/projects/${encodeURIComponent(projectId)}/runs`,
    method: 'POST',
    body: payload
  })
}

export function pauseWorkflowRun(runId: string): Promise<WorkflowRunResponse> {
  return window.hermesDesktop.api<WorkflowRunResponse>({
    path: `/api/workflows/runs/${encodeURIComponent(runId)}/pause`,
    method: 'POST'
  })
}

export function resumeWorkflowRun(runId: string): Promise<WorkflowRunResponse> {
  return window.hermesDesktop.api<WorkflowRunResponse>({
    path: `/api/workflows/runs/${encodeURIComponent(runId)}/resume`,
    method: 'POST'
  })
}

export function stopWorkflowRun(runId: string): Promise<WorkflowRunResponse> {
  return window.hermesDesktop.api<WorkflowRunResponse>({
    path: `/api/workflows/runs/${encodeURIComponent(runId)}/stop`,
    method: 'POST'
  })
}

export function confirmWorkflowNode(runId: string, nodeId: string): Promise<WorkflowRunResponse> {
  return window.hermesDesktop.api<WorkflowRunResponse>({
    path: `/api/workflows/runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/confirm`,
    method: 'POST'
  })
}

export function retryWorkflowNode(runId: string, nodeId: string): Promise<WorkflowRunResponse> {
  return window.hermesDesktop.api<WorkflowRunResponse>({
    path: `/api/workflows/runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/retry`,
    method: 'POST'
  })
}

export function skipWorkflowNode(runId: string, nodeId: string): Promise<WorkflowRunResponse> {
  return window.hermesDesktop.api<WorkflowRunResponse>({
    path: `/api/workflows/runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/skip`,
    method: 'POST'
  })
}

export function sendWorkflowChat(payload: {
  attachments?: string[]
  nodeId?: string | null
  projectId: string
  skillIds?: string[]
  text: string
}): Promise<{ ok: boolean; patch: Record<string, unknown>; reply?: string; target: string | null }> {
  return window.hermesDesktop.api<{ ok: boolean; patch: Record<string, unknown>; reply?: string; target: string | null }>({
    path: `/api/workflows/projects/${encodeURIComponent(payload.projectId)}/chat`,
    method: 'POST',
    body: {
      attachments: payload.attachments ?? [],
      nodeId: payload.nodeId ?? null,
      skillIds: payload.skillIds ?? [],
      text: payload.text
    }
  })
}

export function getWorkflowSlashCommands(projectId: string, query = ''): Promise<{ items: WorkflowSlashCommandItem[] }> {
  const suffix = query ? `?q=${encodeURIComponent(query)}` : ''

  return window.hermesDesktop.api<{ items: WorkflowSlashCommandItem[] }>({
    path: `/api/workflows/projects/${encodeURIComponent(projectId)}/composer/slash${suffix}`
  })
}

export function executeWorkflowSlashCommand(
  projectId: string,
  payload: { command: string; nodeId?: string | null }
): Promise<{ ok: boolean; output: string }> {
  return window.hermesDesktop.api<{ ok: boolean; output: string }>({
    path: `/api/workflows/projects/${encodeURIComponent(projectId)}/composer/slash`,
    method: 'POST',
    body: { command: payload.command, nodeId: payload.nodeId ?? null }
  })
}

export function completeWorkflowComposer(
  projectId: string,
  payload: { cursor?: number; cwd?: string; text: string }
): Promise<{ items: WorkflowComposerCompletionItem[] }> {
  return window.hermesDesktop.api<{ items: WorkflowComposerCompletionItem[] }>({
    path: `/api/workflows/projects/${encodeURIComponent(projectId)}/composer/complete`,
    method: 'POST',
    body: payload
  })
}

export function attachWorkflowComposerFiles(
  projectId: string,
  paths: string[]
): Promise<{ attachments: ReferenceItem[]; ok: boolean; references: ReferenceItem[] }> {
  return window.hermesDesktop.api<{ attachments: ReferenceItem[]; ok: boolean; references: ReferenceItem[] }>({
    path: `/api/workflows/projects/${encodeURIComponent(projectId)}/composer/attachments`,
    method: 'POST',
    body: { paths }
  })
}

export function listWorkflowEvents(projectId: string, since?: number): Promise<WorkflowEventsResponse> {
  const query = typeof since === 'number' ? `?since=${encodeURIComponent(String(since))}` : ''

  return window.hermesDesktop.api<WorkflowEventsResponse>({
    path: `/api/workflows/projects/${encodeURIComponent(projectId)}/events${query}`
  })
}

export function updateWorkflowReferences(projectId: string, references: ReferenceItem[]): Promise<ProjectBundle> {
  return window.hermesDesktop.api<ProjectBundle>({
    path: `/api/workflows/projects/${encodeURIComponent(projectId)}/references`,
    method: 'PUT',
    body: { references }
  })
}

export function updateWorkflowSkills(projectId: string, skills: SkillBinding[]): Promise<ProjectBundle> {
  return window.hermesDesktop.api<ProjectBundle>({
    path: `/api/workflows/projects/${encodeURIComponent(projectId)}/skills`,
    method: 'PUT',
    body: { skills }
  })
}

export function createWorkflowSnapshot(projectId: string): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    path: `/api/workflows/projects/${encodeURIComponent(projectId)}/snapshots`,
    method: 'POST'
  })
}

export function getWorkflowFiles(projectId: string): Promise<WorkflowFilesResponse> {
  return window.hermesDesktop.api<WorkflowFilesResponse>({
    path: `/api/workflows/projects/${encodeURIComponent(projectId)}/files`
  })
}

export function deleteEnvVar(key: string): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    path: '/api/env',
    method: 'DELETE',
    body: { key }
  })
}

export function revealEnvVar(key: string): Promise<{ key: string; value: string }> {
  return window.hermesDesktop.api<{ key: string; value: string }>({
    path: '/api/env/reveal',
    method: 'POST',
    body: { key }
  })
}

export function listOAuthProviders(): Promise<OAuthProvidersResponse> {
  return window.hermesDesktop.api<OAuthProvidersResponse>({
    path: '/api/providers/oauth'
  })
}

export function startOAuthLogin(providerId: string): Promise<OAuthStartResponse> {
  return window.hermesDesktop.api<OAuthStartResponse>({
    path: `/api/providers/oauth/${encodeURIComponent(providerId)}/start`,
    method: 'POST',
    body: {}
  })
}

export function submitOAuthCode(providerId: string, sessionId: string, code: string): Promise<OAuthSubmitResponse> {
  return window.hermesDesktop.api<OAuthSubmitResponse>({
    path: `/api/providers/oauth/${encodeURIComponent(providerId)}/submit`,
    method: 'POST',
    body: { session_id: sessionId, code }
  })
}

export function pollOAuthSession(providerId: string, sessionId: string): Promise<OAuthPollResponse> {
  return window.hermesDesktop.api<OAuthPollResponse>({
    path: `/api/providers/oauth/${encodeURIComponent(providerId)}/poll/${encodeURIComponent(sessionId)}`
  })
}

export function cancelOAuthSession(sessionId: string): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    path: `/api/providers/oauth/sessions/${encodeURIComponent(sessionId)}`,
    method: 'DELETE'
  })
}

export function getSkills(): Promise<SkillInfo[]> {
  return window.hermesDesktop.api<SkillInfo[]>({
    path: '/api/skills'
  })
}

export function toggleSkill(name: string, enabled: boolean): Promise<{ ok: boolean; name: string; enabled: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean; name: string; enabled: boolean }>({
    path: '/api/skills/toggle',
    method: 'PUT',
    body: { name, enabled }
  })
}

export function getToolsets(): Promise<ToolsetInfo[]> {
  return window.hermesDesktop.api<ToolsetInfo[]>({
    path: '/api/tools/toolsets'
  })
}

export function toggleToolset(
  name: string,
  enabled: boolean
): Promise<{ ok: boolean; name: string; enabled: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean; name: string; enabled: boolean }>({
    path: `/api/tools/toolsets/${encodeURIComponent(name)}`,
    method: 'PUT',
    body: { enabled }
  })
}

export function getToolsetConfig(name: string): Promise<ToolsetConfig> {
  return window.hermesDesktop.api<ToolsetConfig>({
    path: `/api/tools/toolsets/${encodeURIComponent(name)}/config`
  })
}

export function selectToolsetProvider(
  name: string,
  provider: string
): Promise<{ ok: boolean; name: string; provider: string }> {
  return window.hermesDesktop.api<{ ok: boolean; name: string; provider: string }>({
    path: `/api/tools/toolsets/${encodeURIComponent(name)}/provider`,
    method: 'PUT',
    body: { provider }
  })
}

export function getMessagingPlatforms(): Promise<MessagingPlatformsResponse> {
  return window.hermesDesktop.api<MessagingPlatformsResponse>({
    path: '/api/messaging/platforms'
  })
}

export function updateMessagingPlatform(
  platformId: string,
  body: MessagingPlatformUpdate
): Promise<{ ok: boolean; platform: string }> {
  return window.hermesDesktop.api<{ ok: boolean; platform: string }>({
    path: `/api/messaging/platforms/${encodeURIComponent(platformId)}`,
    method: 'PUT',
    body
  })
}

export function testMessagingPlatform(platformId: string): Promise<MessagingPlatformTestResponse> {
  return window.hermesDesktop.api<MessagingPlatformTestResponse>({
    path: `/api/messaging/platforms/${encodeURIComponent(platformId)}/test`,
    method: 'POST'
  })
}

export function getCronJobs(): Promise<CronJob[]> {
  return window.hermesDesktop.api<CronJob[]>({
    path: '/api/cron/jobs'
  })
}

export function getCronJob(jobId: string): Promise<CronJob> {
  return window.hermesDesktop.api<CronJob>({
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}`
  })
}

export function createCronJob(body: CronJobCreatePayload): Promise<CronJob> {
  return window.hermesDesktop.api<CronJob>({
    path: '/api/cron/jobs',
    method: 'POST',
    body
  })
}

export function updateCronJob(jobId: string, updates: CronJobUpdates): Promise<CronJob> {
  return window.hermesDesktop.api<CronJob>({
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}`,
    method: 'PUT',
    body: { updates }
  })
}

export function pauseCronJob(jobId: string): Promise<CronJob> {
  return window.hermesDesktop.api<CronJob>({
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}/pause`,
    method: 'POST'
  })
}

export function resumeCronJob(jobId: string): Promise<CronJob> {
  return window.hermesDesktop.api<CronJob>({
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}/resume`,
    method: 'POST'
  })
}

export function triggerCronJob(jobId: string): Promise<CronJob> {
  return window.hermesDesktop.api<CronJob>({
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}/trigger`,
    method: 'POST'
  })
}

export function deleteCronJob(jobId: string): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}`,
    method: 'DELETE'
  })
}

export function getProfiles(): Promise<ProfilesResponse> {
  return window.hermesDesktop.api<ProfilesResponse>({
    path: '/api/profiles'
  })
}

export function createProfile(body: ProfileCreatePayload): Promise<{ name: string; ok: boolean; path: string }> {
  return window.hermesDesktop.api<{ name: string; ok: boolean; path: string }>({
    path: '/api/profiles',
    method: 'POST',
    body
  })
}

export function renameProfile(name: string, newName: string): Promise<{ name: string; ok: boolean; path: string }> {
  return window.hermesDesktop.api<{ name: string; ok: boolean; path: string }>({
    path: `/api/profiles/${encodeURIComponent(name)}`,
    method: 'PATCH',
    body: { new_name: newName }
  })
}

export function deleteProfile(name: string): Promise<{ ok: boolean; path: string }> {
  return window.hermesDesktop.api<{ ok: boolean; path: string }>({
    path: `/api/profiles/${encodeURIComponent(name)}`,
    method: 'DELETE'
  })
}

export function getProfileSoul(name: string): Promise<ProfileSoul> {
  return window.hermesDesktop.api<ProfileSoul>({
    path: `/api/profiles/${encodeURIComponent(name)}/soul`
  })
}

export function updateProfileSoul(name: string, content: string): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    path: `/api/profiles/${encodeURIComponent(name)}/soul`,
    method: 'PUT',
    body: { content }
  })
}

export function getProfileSetupCommand(name: string): Promise<ProfileSetupCommand> {
  return window.hermesDesktop.api<ProfileSetupCommand>({
    path: `/api/profiles/${encodeURIComponent(name)}/setup-command`
  })
}

export function getUsageAnalytics(days = 30): Promise<AnalyticsResponse> {
  return window.hermesDesktop.api<AnalyticsResponse>({
    path: `/api/analytics/usage?days=${Math.max(1, Math.floor(days))}`
  })
}

export function getGlobalModelOptions(): Promise<ModelOptionsResponse> {
  return window.hermesDesktop.api<ModelOptionsResponse>({
    path: '/api/model/options'
  })
}

export interface RecommendedDefaultModel {
  provider: string
  model: string
  /** True/false for Nous (free vs paid tier); null for other providers. */
  free_tier: boolean | null
}

// Recommended default model for a freshly-authenticated provider. Mirrors the
// curation `hermes model` does — for Nous it honors the free/paid tier so a
// free user gets a free model instead of a paid default.
export function getRecommendedDefaultModel(provider: string): Promise<RecommendedDefaultModel> {
  return window.hermesDesktop.api<RecommendedDefaultModel>({
    path: `/api/model/recommended-default?provider=${encodeURIComponent(provider)}`
  })
}

export function setGlobalModel(
  provider: string,
  model: string
): Promise<{ ok: boolean; provider: string; model: string }> {
  return window.hermesDesktop.api<{ ok: boolean; provider: string; model: string }>({
    path: '/api/model/set',
    method: 'POST',
    body: {
      scope: 'main',
      provider,
      model
    }
  })
}

export function getAuxiliaryModels(): Promise<AuxiliaryModelsResponse> {
  return window.hermesDesktop.api<AuxiliaryModelsResponse>({
    path: '/api/model/auxiliary'
  })
}

export function setModelAssignment(body: ModelAssignmentRequest): Promise<ModelAssignmentResponse> {
  return window.hermesDesktop.api<ModelAssignmentResponse>({
    path: '/api/model/set',
    method: 'POST',
    body
  })
}

export function restartGateway(): Promise<ActionResponse> {
  return window.hermesDesktop.api<ActionResponse>({
    path: '/api/gateway/restart',
    method: 'POST'
  })
}

export function updateHermes(): Promise<ActionResponse> {
  return window.hermesDesktop.api<ActionResponse>({
    path: '/api/hermes/update',
    method: 'POST'
  })
}

export function getActionStatus(name: string, lines = 200): Promise<ActionStatusResponse> {
  return window.hermesDesktop.api<ActionStatusResponse>({
    path: `/api/actions/${encodeURIComponent(name)}/status?lines=${Math.max(1, lines)}`
  })
}

export function transcribeAudio(dataUrl: string, mimeType?: string): Promise<AudioTranscriptionResponse> {
  return window.hermesDesktop.api<AudioTranscriptionResponse>({
    path: '/api/audio/transcribe',
    method: 'POST',
    body: {
      data_url: dataUrl,
      mime_type: mimeType
    }
  })
}

export function speakText(text: string): Promise<AudioSpeakResponse> {
  return window.hermesDesktop.api<AudioSpeakResponse>({
    path: '/api/audio/speak',
    method: 'POST',
    body: { text }
  })
}

export function getElevenLabsVoices(): Promise<ElevenLabsVoicesResponse> {
  return window.hermesDesktop.api<ElevenLabsVoicesResponse>({
    path: '/api/audio/elevenlabs/voices'
  })
}
