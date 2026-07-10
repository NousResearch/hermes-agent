import { JsonRpcGatewayClient } from '@hermes/shared'

import type {
  ActionResponse,
  ActionStatusResponse,
  AnalyticsResponse,
  AudioSpeakResponse,
  AudioTranscriptionResponse,
  AuxiliaryModelsResponse,
  BackendUpdateCheckResponse,
  ComputerUseStatus,
  ConfigSchemaResponse,
  CronJob,
  CronJobCreatePayload,
  CronJobUpdates,
  CuratorStatusResponse,
  DebugShareResponse,
  ElevenLabsVoicesResponse,
  EnvVarInfo,
  HermesConfig,
  HermesConfigRecord,
  LogsResponse,
  McpCatalogResponse,
  McpServerSummary,
  MemoryProviderConfig,
  MemoryProviderOAuthStatus,
  MemoryStatusResponse,
  MessagingPlatformsResponse,
  MessagingPlatformTestResponse,
  MessagingPlatformUpdate,
  MoaConfigResponse,
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
  SessionInfo,
  SessionMessagesResponse,
  SessionSearchResponse,
  SkillHubPreview,
  SkillHubScanResult,
  SkillHubSearchResponse,
  SkillHubSourcesResponse,
  SkillInfo,
  StarmapGraph,
  StatusResponse,
  ToolsetConfig,
  ToolsetInfo,
  ToolsetModelsResponse
} from '@/types/hermes'

// Desktop startup fires a burst of read-only data calls (config, profiles,
// model info/options, cron) the moment the backend passes readiness. On a
// profile-heavy or remote install these can each take tens of seconds — e.g.
// /api/profiles runs list_profiles(), which does a recursive skill-tree walk
// per profile — so the 15s default (DEFAULT_FETCH_TIMEOUT_MS in hardening.ts)
// times out a backend that is alive-but-busy, surfacing as a spurious
// "Timed out connecting to Hermes backend" that hangs the UI (#48504).
//
// Give the boot burst a generous per-call timeout instead of raising the
// global default: interactive/runtime calls and the liveness poll (/api/status)
// keep the short default so a genuinely-dead backend is still detected fast.
export const STARTUP_REQUEST_TIMEOUT_MS = 60_000
const DEFAULT_GATEWAY_REQUEST_TIMEOUT_MS = 30_000
const SESSION_LIST_REQUEST_TIMEOUT_MS = 60_000
// prompt.submit is effectively fire-and-forget: turn completion is signaled by
// stream / message.complete events, NOT by the RPC return. A long turn (MoA
// presets running references + aggregator in series, deep reasoning, large tool
// chains) can legitimately take minutes to ACK, so bounding the ack by the
// generic 30s default surfaces a false "request timed out" toast while the turn
// is still running and will succeed (issue #55024). Match the backend's
// agent-turn ceiling (agent.gateway_timeout = 1800s) so the ack timeout only
// ever fires when the turn itself would have been abandoned server-side.
export const PROMPT_SUBMIT_REQUEST_TIMEOUT_MS = 1_800_000

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
  BackendUpdateCheckResponse,
  ComputerUseCheck,
  ComputerUsePermissionSource,
  ComputerUseStatus,
  ConfigFieldSchema,
  ConfigSchemaResponse,
  CronJob,
  CronJobCreatePayload,
  CronJobSchedule,
  CronJobUpdates,
  CuratorStatusResponse,
  DebugShareResponse,
  ElevenLabsVoice,
  ElevenLabsVoicesResponse,
  EnvVarInfo,
  GatewayReadyPayload,
  HermesConfig,
  HermesConfigRecord,
  LogsResponse,
  McpCatalogEntry,
  McpCatalogResponse,
  McpServerSummary,
  McpServerTestResponse,
  MemoryProviderConfig,
  MemoryProviderOAuthStatus,
  MemoryStatusResponse,
  MessagingEnvVarInfo,
  MessagingHomeChannel,
  MessagingPlatformInfo,
  MessagingPlatformsResponse,
  MessagingPlatformTestResponse,
  MessagingPlatformUpdate,
  MoaConfigResponse,
  MoaModelSlot,
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
  ProjectFolder,
  ProjectInfo,
  ProjectsPayload,
  RpcEvent,
  SessionCreateResponse,
  SessionInfo,
  SessionMessage,
  SessionMessagesResponse,
  SessionResumeResponse,
  SessionRuntimeInfo,
  SessionSearchResponse,
  SessionSearchResult,
  SkillHubInstalledEntry,
  SkillHubPreview,
  SkillHubResult,
  SkillHubScanResult,
  SkillHubSearchResponse,
  SkillHubSource,
  SkillHubSourcesResponse,
  SkillInfo,
  StaleAuxAssignment,
  StarmapGraph,
  StatusResponse,
  ToolsetConfig,
  ToolsetInfo,
  ToolsetModel,
  ToolsetModelsResponse
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

// Profile that profile-scoped REST settings (config/env/skills/tools/model/…)
// should target. Mirrors $activeGatewayProfile, pushed in from the store via
// setApiRequestProfile so this module needs no store import (avoids a cycle).
// Electron main consumes request.profile to pick which backend *process* serves
// the call; each pooled backend already has its own HERMES_HOME, so no backend
// change is needed. Null → primary, so single-profile users are unaffected.
let _apiProfile: null | string = null

export function setApiRequestProfile(profile: null | string): void {
  _apiProfile = profile || null
}

function profileScoped(): { profile?: string } {
  return _apiProfile ? { profile: _apiProfile } : {}
}

export async function listSessions(
  limit = 40,
  minMessages = 0,
  archived: 'exclude' | 'include' | 'only' = 'exclude',
  order: 'created' | 'recent' = 'recent'
): Promise<PaginatedSessions> {
  const result = await window.hermesDesktop.api<PaginatedSessions>({
    path:
      `/api/sessions?limit=${limit}&offset=0&min_messages=${Math.max(0, minMessages)}` +
      `&archived=${archived}&order=${order}`,
    timeoutMs: SESSION_LIST_REQUEST_TIMEOUT_MS
  })

  return {
    ...result,
    sessions: result.sessions.slice(0, limit),
    offset: 0
  }
}

// Unified, read-only session list aggregated across ALL profiles. Served by the
// primary backend straight off each profile's state.db — no per-profile backend
// is spawned. Single-profile users get the same rows as listSessions(), tagged
// profile="default".
// Source scoping lets callers split the unified list into independent slices:
// recents pass `excludeSources: ['cron']`, the cron-jobs section passes
// `source: 'cron'`. Without this a burst of (always-newest) cron sessions
// consumes the whole recents page and starves real conversations.
export interface SessionSourceFilter {
  source?: string
  excludeSources?: string[]
}

export async function listAllProfileSessions(
  limit = 40,
  minMessages = 0,
  archived: 'exclude' | 'include' | 'only' = 'exclude',
  order: 'created' | 'recent' = 'recent',
  profile: 'all' | (string & {}) = 'all',
  filter: SessionSourceFilter = {}
): Promise<PaginatedSessions> {
  const sourceParam = filter.source ? `&source=${encodeURIComponent(filter.source)}` : ''

  const excludeParam = filter.excludeSources?.length
    ? `&exclude_sources=${encodeURIComponent(filter.excludeSources.join(','))}`
    : ''

  const result = await window.hermesDesktop.api<PaginatedSessions>({
    path:
      `/api/profiles/sessions?limit=${limit}&offset=0&min_messages=${Math.max(0, minMessages)}` +
      `&archived=${archived}&order=${order}&profile=${encodeURIComponent(profile)}${sourceParam}${excludeParam}`,
    timeoutMs: SESSION_LIST_REQUEST_TIMEOUT_MS
  })

  return {
    ...result,
    sessions: result.sessions.slice(0, limit),
    offset: 0
  }
}

// Mutations take the owning `profile` so Electron routes them to that profile's
// backend (remote pool or local primary) via request.profile — matching the
// read path. A remote session's row lives only on its remote host, so a mutation
// that hit the local primary would no-op or 404. Omit for the current/default.
export function setSessionArchived(id: string, archived: boolean, profile?: string | null): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    ...(profile ? { profile } : {}),
    path: `/api/sessions/${encodeURIComponent(id)}`,
    method: 'PATCH',
    body: { archived }
  })
}

export function searchSessions(query: string): Promise<SessionSearchResponse> {
  return window.hermesDesktop.api<SessionSearchResponse>({
    path: `/api/sessions/search?q=${encodeURIComponent(query)}`
  })
}

// Resolves a single session row by id on one backend (the active profile, or
// the given `profile`). The backend resolves exact ids and unique prefixes and
// 404s when the id isn't on that profile — so a cheap by-id lookup replaces the
// cross-profile list scan when locating an unknown id's owner.
export function getSession(id: string, profile?: string | null): Promise<SessionInfo> {
  const suffix = profile ? `?profile=${encodeURIComponent(profile)}` : ''

  return window.hermesDesktop.api<SessionInfo>({
    ...(profile ? { profile } : {}),
    path: `/api/sessions/${encodeURIComponent(id)}${suffix}`
  })
}

// Reads another profile's transcript. For a remote profile Electron reroutes
// this GET to the remote backend (which serves its own state.db); for a local
// profile the primary opens that profile's state.db via ?profile=. Omit for
// the current/default profile.
export function getSessionMessages(id: string, profile?: string | null): Promise<SessionMessagesResponse> {
  const suffix = profile ? `?profile=${encodeURIComponent(profile)}` : ''

  return window.hermesDesktop.api<SessionMessagesResponse>({
    ...(profile ? { profile } : {}),
    path: `/api/sessions/${encodeURIComponent(id)}/messages${suffix}`
  })
}

export function deleteSession(id: string, profile?: string | null): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    ...(profile ? { profile } : {}),
    path: `/api/sessions/${encodeURIComponent(id)}`,
    method: 'DELETE'
  })
}

export function renameSession(
  id: string,
  title: string,
  profile?: string | null
): Promise<{ ok: boolean; title: string }> {
  return window.hermesDesktop.api<{ ok: boolean; title: string }>({
    ...(profile ? { profile } : {}),
    path: `/api/sessions/${encodeURIComponent(id)}`,
    method: 'PATCH',
    body: { title, ...(profile ? { profile } : {}) }
  })
}

export function getGlobalModelInfo(): Promise<ModelInfoResponse> {
  return window.hermesDesktop.api<ModelInfoResponse>({
    ...profileScoped(),
    path: '/api/model/info',
    timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
  })
}

export function getStatus(): Promise<StatusResponse> {
  return window.hermesDesktop.api<StatusResponse>({
    ...profileScoped(),
    path: '/api/status'
  })
}

export function getLogs(params: {
  component?: string
  file?: string
  level?: string
  lines?: number
  search?: string
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

  if (params.search) {
    query.set('search', params.search)
  }

  const suffix = query.toString()

  return window.hermesDesktop.api<LogsResponse>({
    ...profileScoped(),
    path: suffix ? `/api/logs?${suffix}` : '/api/logs'
  })
}

export function getHermesConfig(): Promise<HermesConfig> {
  return window.hermesDesktop.api<HermesConfig>({
    ...profileScoped(),
    path: '/api/config',
    timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
  })
}

export function getHermesConfigRecord(): Promise<HermesConfigRecord> {
  return window.hermesDesktop.api<HermesConfigRecord>({
    ...profileScoped(),
    path: '/api/config'
  })
}

export function getHermesConfigDefaults(): Promise<HermesConfigRecord> {
  return window.hermesDesktop.api<HermesConfigRecord>({
    ...profileScoped(),
    path: '/api/config/defaults',
    timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
  })
}

export function getHermesConfigSchema(): Promise<ConfigSchemaResponse> {
  return window.hermesDesktop.api<ConfigSchemaResponse>({
    ...profileScoped(),
    path: '/api/config/schema'
  })
}

export function saveHermesConfig(config: HermesConfigRecord): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    ...profileScoped(),
    path: '/api/config',
    method: 'PUT',
    body: { config }
  })
}

export function getMemoryProviderConfig(provider: string): Promise<MemoryProviderConfig> {
  return window.hermesDesktop.api<MemoryProviderConfig>({
    path: `/api/memory/providers/${encodeURIComponent(provider)}/config`
  })
}

export function saveMemoryProviderConfig(provider: string, values: Record<string, string>): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    path: `/api/memory/providers/${encodeURIComponent(provider)}/config`,
    method: 'PUT',
    body: { values }
  })
}

export function getEnvVars(): Promise<Record<string, EnvVarInfo>> {
  return window.hermesDesktop.api<Record<string, EnvVarInfo>>({
    ...profileScoped(),
    path: '/api/env'
  })
}

export function setEnvVar(key: string, value: string): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    ...profileScoped(),
    path: '/api/env',
    method: 'PUT',
    body: { key, value }
  })
}

export function validateProviderCredential(
  key: string,
  value: string,
  apiKey?: string
): Promise<{ ok: boolean; reachable: boolean; message: string; models?: string[] }> {
  return window.hermesDesktop.api<{ ok: boolean; reachable: boolean; message: string; models?: string[] }>({
    ...profileScoped(),
    path: '/api/providers/validate',
    method: 'POST',
    body: { key, value, api_key: apiKey ?? '' }
  })
}

export function deleteEnvVar(key: string): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    ...profileScoped(),
    path: '/api/env',
    method: 'DELETE',
    body: { key }
  })
}

export function revealEnvVar(key: string): Promise<{ key: string; value: string }> {
  return window.hermesDesktop.api<{ key: string; value: string }>({
    ...profileScoped(),
    path: '/api/env/reveal',
    method: 'POST',
    body: { key }
  })
}

export function listOAuthProviders(): Promise<OAuthProvidersResponse> {
  return window.hermesDesktop.api<OAuthProvidersResponse>({
    ...profileScoped(),
    path: '/api/providers/oauth'
  })
}

export function disconnectOAuthProvider(providerId: string): Promise<{ ok: boolean; provider: string }> {
  return window.hermesDesktop.api<{ ok: boolean; provider: string }>({
    ...profileScoped(),
    path: `/api/providers/oauth/${encodeURIComponent(providerId)}`,
    method: 'DELETE'
  })
}

export function startOAuthLogin(providerId: string): Promise<OAuthStartResponse> {
  return window.hermesDesktop.api<OAuthStartResponse>({
    ...profileScoped(),
    path: `/api/providers/oauth/${encodeURIComponent(providerId)}/start`,
    method: 'POST',
    body: {}
  })
}

export function submitOAuthCode(providerId: string, sessionId: string, code: string): Promise<OAuthSubmitResponse> {
  return window.hermesDesktop.api<OAuthSubmitResponse>({
    ...profileScoped(),
    path: `/api/providers/oauth/${encodeURIComponent(providerId)}/submit`,
    method: 'POST',
    body: { session_id: sessionId, code }
  })
}

export function pollOAuthSession(providerId: string, sessionId: string): Promise<OAuthPollResponse> {
  return window.hermesDesktop.api<OAuthPollResponse>({
    ...profileScoped(),
    path: `/api/providers/oauth/${encodeURIComponent(providerId)}/poll/${encodeURIComponent(sessionId)}`
  })
}

export function cancelOAuthSession(sessionId: string): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    ...profileScoped(),
    path: `/api/providers/oauth/sessions/${encodeURIComponent(sessionId)}`,
    method: 'DELETE'
  })
}

// Memory-provider OAuth connect (provider-keyed; 404s for providers without an
// OAuth flow). Profile-scoped: the grant lands in the active profile's config.
export function startMemoryProviderOAuth(provider: string): Promise<MemoryProviderOAuthStatus> {
  return window.hermesDesktop.api<MemoryProviderOAuthStatus>({
    ...profileScoped(),
    path: `/api/memory/providers/${encodeURIComponent(provider)}/oauth/start`,
    method: 'POST'
  })
}

export function getMemoryProviderOAuthStatus(provider: string): Promise<MemoryProviderOAuthStatus> {
  return window.hermesDesktop.api<MemoryProviderOAuthStatus>({
    ...profileScoped(),
    path: `/api/memory/providers/${encodeURIComponent(provider)}/oauth/status`
  })
}

export function getSkills(): Promise<SkillInfo[]> {
  return window.hermesDesktop.api<SkillInfo[]>({
    ...profileScoped(),
    path: '/api/skills'
  })
}

export function getStarmapGraph(): Promise<StarmapGraph> {
  return window.hermesDesktop.api<StarmapGraph>({
    ...profileScoped(),
    // Backend REST contract — stays /api/learning even though the UI feature is
    // now "star map". Renaming this would break against an un-upgraded backend.
    path: '/api/learning/graph'
  })
}

export interface LearningNodeDetail {
  content: string
  kind: 'memory' | 'skill'
  label: string
  ok: boolean
}

export function getLearningNode(id: string): Promise<LearningNodeDetail> {
  return window.hermesDesktop.api<LearningNodeDetail>({
    ...profileScoped(),
    path: `/api/learning/node?id=${encodeURIComponent(id)}`
  })
}

export function deleteLearningNode(id: string): Promise<{ message: string; ok: boolean }> {
  return window.hermesDesktop.api<{ message: string; ok: boolean }>({
    ...profileScoped(),
    path: '/api/learning/node',
    method: 'DELETE',
    body: { id }
  })
}

export function editLearningNode(id: string, content: string): Promise<{ message: string; ok: boolean }> {
  return window.hermesDesktop.api<{ message: string; ok: boolean }>({
    ...profileScoped(),
    path: '/api/learning/node',
    method: 'PUT',
    body: { content, id }
  })
}

export function toggleSkill(name: string, enabled: boolean): Promise<{ ok: boolean; name: string; enabled: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean; name: string; enabled: boolean }>({
    ...profileScoped(),
    path: '/api/skills/toggle',
    method: 'PUT',
    body: { name, enabled }
  })
}

export interface McpTestResult {
  ok: boolean
  error?: string
  tools: { name: string; description: string }[]
  /** Capability counts (absent on older backends / failed probes). */
  prompts?: number
  resources?: number
}

/** Connect to the server, list its tools, disconnect. Slow (spawns/handshakes
 *  for real) — well past the 15s default fetch timeout. */
export function testMcpServer(name: string): Promise<McpTestResult> {
  return window.hermesDesktop.api<McpTestResult>({
    ...profileScoped(),
    path: `/api/mcp/servers/${encodeURIComponent(name)}/test`,
    method: 'POST',
    timeoutMs: 60_000
  })
}

/** Replace the whole `mcp_servers` map (the mcp.json editor's save). Unlike
 *  `saveHermesConfig`, this REPLACES rather than deep-merges, so deletes,
 *  re-enables (dropping `enabled: false`), and removed nested fields persist. */
export function saveMcpServers(servers: Record<string, Record<string, unknown>>): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    ...profileScoped(),
    path: '/api/mcp/servers',
    method: 'PUT',
    body: { servers }
  })
}

/** Run the OAuth flow for an HTTP server — opens the system browser and blocks
 *  until the user finishes (or gives up), hence the very generous timeout. */
export function authMcpServer(name: string): Promise<McpTestResult> {
  return window.hermesDesktop.api<McpTestResult>({
    ...profileScoped(),
    path: `/api/mcp/servers/${encodeURIComponent(name)}/auth`,
    method: 'POST',
    timeoutMs: 300_000
  })
}

export function getToolsets(): Promise<ToolsetInfo[]> {
  return window.hermesDesktop.api<ToolsetInfo[]>({
    ...profileScoped(),
    path: '/api/tools/toolsets'
  })
}

export function toggleToolset(
  name: string,
  enabled: boolean
): Promise<{ ok: boolean; name: string; enabled: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean; name: string; enabled: boolean }>({
    ...profileScoped(),
    path: `/api/tools/toolsets/${encodeURIComponent(name)}`,
    method: 'PUT',
    body: { enabled }
  })
}

export function getToolsetConfig(name: string): Promise<ToolsetConfig> {
  return window.hermesDesktop.api<ToolsetConfig>({
    ...profileScoped(),
    path: `/api/tools/toolsets/${encodeURIComponent(name)}/config`
  })
}

export function getToolsetModels(name: string, provider?: string): Promise<ToolsetModelsResponse> {
  const suffix = provider ? `?provider=${encodeURIComponent(provider)}` : ''

  return window.hermesDesktop.api<ToolsetModelsResponse>({
    ...profileScoped(),
    path: `/api/tools/toolsets/${encodeURIComponent(name)}/models${suffix}`
  })
}

export function selectToolsetModel(
  name: string,
  model: string,
  provider?: string
): Promise<{ ok: boolean; name: string; model: string }> {
  return window.hermesDesktop.api<{ ok: boolean; name: string; model: string }>({
    ...profileScoped(),
    path: `/api/tools/toolsets/${encodeURIComponent(name)}/model`,
    method: 'PUT',
    body: { model, provider }
  })
}

export function selectToolsetProvider(
  name: string,
  provider: string
): Promise<{ ok: boolean; name: string; provider: string }> {
  return window.hermesDesktop.api<{ ok: boolean; name: string; provider: string }>({
    ...profileScoped(),
    path: `/api/tools/toolsets/${encodeURIComponent(name)}/provider`,
    method: 'PUT',
    body: { provider }
  })
}

export function runToolsetPostSetup(name: string, key: string): Promise<ActionResponse & { key: string }> {
  return window.hermesDesktop.api<ActionResponse & { key: string }>({
    ...profileScoped(),
    path: `/api/tools/toolsets/${encodeURIComponent(name)}/post-setup`,
    method: 'POST',
    body: { key }
  })
}

export function getComputerUseStatus(): Promise<ComputerUseStatus> {
  return window.hermesDesktop.api<ComputerUseStatus>({
    ...profileScoped(),
    path: '/api/tools/computer-use/status'
  })
}

export function grantComputerUsePermissions(): Promise<ActionResponse> {
  return window.hermesDesktop.api<ActionResponse>({
    ...profileScoped(),
    path: '/api/tools/computer-use/permissions/grant',
    method: 'POST'
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

export function getCronJobs(profile = 'default'): Promise<CronJob[]> {
  return window.hermesDesktop.api<CronJob[]>({
    path: `/api/cron/jobs?profile=${encodeURIComponent(profile)}`,
    timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
  })
}

export function getCronJob(jobId: string): Promise<CronJob> {
  return window.hermesDesktop.api<CronJob>({
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}`
  })
}

export async function getCronJobRuns(jobId: string, limit = 20): Promise<SessionInfo[]> {
  const { runs } = await window.hermesDesktop.api<{ runs: SessionInfo[] }>({
    path: `/api/cron/jobs/${encodeURIComponent(jobId)}/runs?limit=${limit}`
  })

  return runs ?? []
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
    path: '/api/profiles',
    timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
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
    ...profileScoped(),
    path: `/api/analytics/usage?days=${Math.max(1, Math.floor(days))}`
  })
}

export function getGlobalModelOptions(opts?: {
  refresh?: boolean
  includeUnconfigured?: boolean
  explicitOnly?: boolean
}): Promise<ModelOptionsResponse> {
  const params = new URLSearchParams()

  if (opts?.refresh) {
    params.set('refresh', '1')
  }

  if (opts?.includeUnconfigured) {
    params.set('include_unconfigured', '1')
  }

  if (opts?.explicitOnly !== false) {
    params.set('explicit_only', '1')
  }

  return window.hermesDesktop.api<ModelOptionsResponse>({
    ...profileScoped(),
    path: params.size > 0 ? `/api/model/options?${params.toString()}` : '/api/model/options',
    timeoutMs: STARTUP_REQUEST_TIMEOUT_MS
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
    ...profileScoped(),
    path: `/api/model/recommended-default?provider=${encodeURIComponent(provider)}`
  })
}

export function setGlobalModel(
  provider: string,
  model: string
): Promise<{ ok: boolean; provider: string; model: string }> {
  return window.hermesDesktop.api<{ ok: boolean; provider: string; model: string }>({
    ...profileScoped(),
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
    ...profileScoped(),
    path: '/api/model/auxiliary'
  })
}

export function getMoaModels(): Promise<MoaConfigResponse> {
  return window.hermesDesktop.api<MoaConfigResponse>({
    ...profileScoped(),
    path: '/api/model/moa'
  })
}

export function saveMoaModels(body: MoaConfigResponse): Promise<MoaConfigResponse & { ok: boolean }> {
  return window.hermesDesktop.api<MoaConfigResponse & { ok: boolean }>({
    ...profileScoped(),
    path: '/api/model/moa',
    method: 'PUT',
    body
  })
}

export function setModelAssignment(body: ModelAssignmentRequest): Promise<ModelAssignmentResponse> {
  return window.hermesDesktop.api<ModelAssignmentResponse>({
    ...profileScoped(),
    path: '/api/model/set',
    method: 'POST',
    body
  })
}

export function restartGateway(): Promise<ActionResponse> {
  return window.hermesDesktop.api<ActionResponse>({
    ...profileScoped(),
    path: '/api/gateway/restart',
    method: 'POST'
  })
}

export function updateHermes(): Promise<ActionResponse> {
  return window.hermesDesktop.api<ActionResponse>({
    ...profileScoped(),
    path: '/api/hermes/update',
    method: 'POST'
  })
}

/** Query the connected backend's own update state. In remote mode this is the
 *  authoritative source for the backend's behind-count + "what's changed",
 *  distinct from the Electron client clone's git state. */
export function checkHermesUpdate(force = false): Promise<BackendUpdateCheckResponse> {
  return window.hermesDesktop.api<BackendUpdateCheckResponse>({
    ...profileScoped(),
    path: `/api/hermes/update/check${force ? '?force=true' : ''}`
  })
}

export function getActionStatus(name: string, lines = 200): Promise<ActionStatusResponse> {
  return window.hermesDesktop.api<ActionStatusResponse>({
    ...profileScoped(),
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

// ---------------------------------------------------------------------------
// Skills hub — search / preview / scan / install (parity with `hermes skills`
// and the dashboard's Browse-hub tab). Installs spawn background actions whose
// logs are tailed via getActionStatus().
// ---------------------------------------------------------------------------

const HUB_REQUEST_TIMEOUT_MS = 45_000

export function getSkillHubSources(): Promise<SkillHubSourcesResponse> {
  return window.hermesDesktop.api<SkillHubSourcesResponse>({
    ...profileScoped(),
    path: '/api/skills/hub/sources',
    timeoutMs: HUB_REQUEST_TIMEOUT_MS
  })
}

export function searchSkillsHub(query: string, source = 'all', limit = 20): Promise<SkillHubSearchResponse> {
  const params = new URLSearchParams({ q: query, source, limit: String(limit) })

  return window.hermesDesktop.api<SkillHubSearchResponse>({
    ...profileScoped(),
    path: `/api/skills/hub/search?${params.toString()}`,
    timeoutMs: HUB_REQUEST_TIMEOUT_MS
  })
}

export function previewSkillHub(identifier: string): Promise<SkillHubPreview> {
  return window.hermesDesktop.api<SkillHubPreview>({
    ...profileScoped(),
    path: `/api/skills/hub/preview?identifier=${encodeURIComponent(identifier)}`,
    timeoutMs: HUB_REQUEST_TIMEOUT_MS
  })
}

export function scanSkillHub(identifier: string): Promise<SkillHubScanResult> {
  return window.hermesDesktop.api<SkillHubScanResult>({
    ...profileScoped(),
    path: `/api/skills/hub/scan?identifier=${encodeURIComponent(identifier)}`,
    timeoutMs: HUB_REQUEST_TIMEOUT_MS
  })
}

export function installSkillFromHub(identifier: string): Promise<ActionResponse> {
  return window.hermesDesktop.api<ActionResponse>({
    ...profileScoped(),
    path: '/api/skills/hub/install',
    method: 'POST',
    body: { identifier }
  })
}

export function uninstallSkillFromHub(name: string): Promise<ActionResponse> {
  return window.hermesDesktop.api<ActionResponse>({
    ...profileScoped(),
    path: '/api/skills/hub/uninstall',
    method: 'POST',
    body: { name }
  })
}

export function updateSkillsFromHub(): Promise<ActionResponse> {
  return window.hermesDesktop.api<ActionResponse>({
    ...profileScoped(),
    path: '/api/skills/hub/update',
    method: 'POST',
    body: {}
  })
}

// ---------------------------------------------------------------------------
// MCP servers — structured list / test / enable toggle / catalog (parity with
// `hermes mcp` and the dashboard MCP page). Raw JSON editing stays in
// config.yaml via saveHermesConfig.
// ---------------------------------------------------------------------------

export function listMcpServers(): Promise<{ servers: McpServerSummary[] }> {
  return window.hermesDesktop.api<{ servers: McpServerSummary[] }>({
    ...profileScoped(),
    path: '/api/mcp/servers'
  })
}

export function setMcpServerEnabled(name: string, enabled: boolean): Promise<{ ok: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean }>({
    ...profileScoped(),
    path: `/api/mcp/servers/${encodeURIComponent(name)}/enabled`,
    method: 'PUT',
    body: { enabled }
  })
}

export function getMcpCatalog(): Promise<McpCatalogResponse> {
  return window.hermesDesktop.api<McpCatalogResponse>({
    ...profileScoped(),
    path: '/api/mcp/catalog'
  })
}

export function installMcpCatalogEntry(
  name: string,
  env: Record<string, string> = {}
): Promise<{ ok: boolean; name?: string; pid?: number; action?: string; background?: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean; name?: string; pid?: number; action?: string; background?: boolean }>({
    ...profileScoped(),
    path: '/api/mcp/catalog/install',
    method: 'POST',
    body: { name, env, enable: true },
    timeoutMs: 60_000
  })
}

// ---------------------------------------------------------------------------
// Memory data + curator (parity with `hermes memory` / `hermes curator`).
// ---------------------------------------------------------------------------

export function getMemoryStatus(): Promise<MemoryStatusResponse> {
  return window.hermesDesktop.api<MemoryStatusResponse>({
    ...profileScoped(),
    path: '/api/memory'
  })
}

export function resetMemory(target: 'all' | 'memory' | 'user'): Promise<{ ok: boolean; deleted: string[] }> {
  return window.hermesDesktop.api<{ ok: boolean; deleted: string[] }>({
    ...profileScoped(),
    path: '/api/memory/reset',
    method: 'POST',
    body: { target }
  })
}

export function getCuratorStatus(): Promise<CuratorStatusResponse> {
  return window.hermesDesktop.api<CuratorStatusResponse>({
    ...profileScoped(),
    path: '/api/curator'
  })
}

export function setCuratorPaused(paused: boolean): Promise<{ ok: boolean; paused: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean; paused: boolean }>({
    ...profileScoped(),
    path: '/api/curator/paused',
    method: 'PUT',
    body: { paused }
  })
}

export function runCurator(): Promise<ActionResponse> {
  return window.hermesDesktop.api<ActionResponse>({
    ...profileScoped(),
    path: '/api/curator/run',
    method: 'POST',
    body: {}
  })
}

// ---------------------------------------------------------------------------
// Maintenance operations (parity with `hermes doctor` / `hermes security
// audit` / `hermes backup` / `hermes debug share` and the dashboard System
// page). All except debug share are spawn-based background actions tailed via
// getActionStatus().
// ---------------------------------------------------------------------------

export function runDoctor(): Promise<ActionResponse> {
  return window.hermesDesktop.api<ActionResponse>({ path: '/api/ops/doctor', method: 'POST', body: {} })
}

export function runSecurityAudit(): Promise<ActionResponse> {
  return window.hermesDesktop.api<ActionResponse>({ path: '/api/ops/security-audit', method: 'POST', body: {} })
}

export function runBackup(): Promise<ActionResponse & { archive?: string }> {
  return window.hermesDesktop.api<ActionResponse & { archive?: string }>({
    path: '/api/ops/backup',
    method: 'POST',
    body: {}
  })
}

export function runDebugShare(): Promise<DebugShareResponse> {
  return window.hermesDesktop.api<DebugShareResponse>({
    path: '/api/ops/debug-share',
    method: 'POST',
    body: {},
    // Synchronous upload of report + logs to the paste service.
    timeoutMs: 120_000
  })
}
