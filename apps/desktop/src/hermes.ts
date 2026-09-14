// The desktop REST/WS client, split by domain under src/api/. This module is
// the compatibility barrel: every helper keeps its historical `@/hermes`
// import path while the implementations live in focused files.
// client is the one module with internals: profileScoped / connectionScoped /
// capabilityScoped are shared across api/ but must not reach call sites, or
// request scoping stops having a single owner.

import { profileScoped as _profileScoped } from './api/client'

export {
  getApiRequestConnection,
  getApiRequestProfile,
  hermesApi,
  HermesGateway,
  profileScopeKey,
  PROMPT_SUBMIT_REQUEST_TIMEOUT_MS,
  setApiRequestConnection,
  setApiRequestProfile,
  STARTUP_REQUEST_TIMEOUT_MS
} from './api/client'
export type { ProfileScope } from './api/client'
export * from './api/config'
export * from './api/cron'
export * from './api/local-models'
export * from './api/mcp'
export * from './api/messaging'
export * from './api/models'
export * from './api/plugins'
export * from './api/profiles'
export * from './api/sessions'
export * from './api/skills'
export * from './api/system'
export * from './api/toolsets'

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
  AudioTtsLeaseResponse,
  AutomationBlueprint,
  AutomationBlueprintField,
  AuxiliaryModelsResponse,
  AuxiliaryTaskAssignment,
  BackendUpdateCheckResponse,
  ComputerUseCheck,
  ComputerUsePermissionSource,
  ComputerUseStatus,
  ConfigFieldSchema,
  ConfigSchemaResponse,
  CronDeliveryTarget,
  CronJob,
  CronJobCreatePayload,
  CronJobSchedule,
  CronJobUpdates,
  CuratorStatusResponse,
  CustomEndpoint,
  CustomEndpointsResponse,
  CustomEndpointUpdate,
  CustomEndpointValidationResponse,
  DebugShareResponse,
  ElevenLabsVoice,
  ElevenLabsVoicesResponse,
  EnvVarInfo,
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
  PaginatedSessions,
  PairingResponse,
  PairingUser,
  ProfileCreatePayload,
  ProfileDesktopOverlay,
  ProfileInfo,
  ProfileSetupCommand,
  ProfileSoul,
  ProfilesResponse,
  ProjectFolder,
  ProjectInfo,
  ProjectsPayload,
  SessionCreateResponse,
  SessionInfo,
  SessionMessage,
  SessionMessagesResponse,
  SessionResumeResult,
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
  TelegramOnboardingApplyResponse,
  TelegramOnboardingStartResponse,
  TelegramOnboardingStatusResponse,
  ToolsetConfig,
  ToolsetInfo,
  ToolsetModel,
  ToolsetModelsResponse,
  WebhookCreatePayload,
  WebhookCreateResponse,
  WebhookEnableResponse,
  WebhookRoute,
  WebhooksResponse
} from '@/types/hermes'

// ── Journey/Learning API ──────────────────────────────────────────────────

export interface LearningRecallDraft {
  connected_count: number
  /** Threat-scan pattern ids matched in the recalled body (empty = clean). */
  findings: string[]
  id: string
  kind: 'memory' | 'skill'
  label: string
  ok: boolean
  text: string
  truncated: boolean
}

export function getLearningRecallDraft(id: string, profile?: string): Promise<LearningRecallDraft> {
  const scope = profile ? { profile } : _profileScoped()

  return window.hermesDesktop.api<LearningRecallDraft>({
    ...scope,
    path: `/api/learning/recall-draft?id=${encodeURIComponent(id)}`
  })
}

/** One raw message from a provider-side session (journey source corpus). */
export interface ProviderSessionMessage {
  content: string
  peer: string
  /** 'user' | 'assistant' when the provider knows which peer is the human. */
  role?: string
  /** Unix seconds, or null when the provider didn't record a time. */
  timestamp: null | number
}

export interface ProviderSessionResponse {
  messages: ProviderSessionMessage[]
  provider: null | string
  session_id: string
}

/** Source corpus behind a provider-contributed journey node — the raw
 *  provider-side conversation a derived fact (e.g. a Honcho conclusion)
 *  came from. Empty `messages` means unavailable, not an error. */
export function getLearningProviderSession(sessionId: string): Promise<ProviderSessionResponse> {
  return window.hermesDesktop.api<ProviderSessionResponse>({
    ..._profileScoped(),
    path: `/api/learning/provider-session?session_id=${encodeURIComponent(sessionId)}`
  })
}

export interface MaterializedProviderSession {
  created: boolean
  message_count: number
  ok: boolean
  provider: null | string
  session_id: string
  title: string
}

/** Recreate a provider-side conversation (journey source corpus) as a real
 *  Hermes session, so it can be read and continued like any other session.
 *  Idempotent: an already-materialized conversation returns `created: false`
 *  with the same session id. */
export function materializeLearningProviderSession(sessionId: string): Promise<MaterializedProviderSession> {
  return window.hermesDesktop.api<MaterializedProviderSession>({
    ..._profileScoped(),
    path: '/api/learning/provider-session/materialize',
    method: 'POST',
    body: { session_id: sessionId }
  })
}
