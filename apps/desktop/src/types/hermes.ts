import type { GatewaySessionRuntimeInfo } from '@hermes/shared/gateway-contracts'

import type {
  ActionResponse,
  ActionStatusResponse,
  AnalyticsDailyEntry,
  AnalyticsModelEntry,
  AnalyticsSkillEntry,
  AnalyticsSkillsSummary,
  AnalyticsToolEntry,
  AnalyticsTotals,
  AudioSpeakResponse,
  AudioTranscriptionResponse,
  AuxiliaryTaskAssignment,
  BackendUpdateCheckResponse,
  BackendUpdateCommit,
  ComputerUseCheck,
  ComputerUsePermissionSource,
  ComputerUseStatus,
  ConfigFieldSchema,
  ConfigSchemaResponse,
  ContextBreakdown,
  ContextUsageCategory,
  CronJob,
  CronJobCreatePayload,
  CronJobSchedule,
  CronJobUpdates,
  CuratorStatusResponse,
  CustomEndpoint,
  CustomEndpointUpdate,
  CustomEndpointValidationResponse,
  DebugShareResponse,
  ElevenLabsVoice,
  ElevenLabsVoicesResponse,
  EnvVarInfo,
  GatewayReadyPayload,
  LogsResponse,
  McpServerSummary,
  MemoryProviderConfig,
  MemoryProviderField,
  MemoryProviderFieldKind,
  MemoryProviderFieldOption,
  MemoryProviderOAuthStatus,
  MessagingEnvVarInfo,
  MessagingHomeChannel,
  MessagingPlatformInfo,
  MessagingPlatformTestResponse,
  MessagingPlatformUpdate,
  MessagingPlatformsResponse,
  MoaModelSlot,
  ModelAssignmentRequest,
  ModelAssignmentResponse,
  ModelCapabilities,
  ModelInfoResponse,
  ModelOptionProvider,
  ModelOptionsResponse,
  ModelPricing,
  OAuthPollResponse,
  OAuthProvider,
  OAuthProviderStatus,
  OAuthProvidersResponse,
  OAuthSubmitResponse,
  PlatformStatus,
  ProfileCreatePayload,
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
  SessionSearchResponse,
  SessionSearchResult,
  SkillHubInstalledEntry,
  SkillHubPreview,
  SkillHubResult,
  SkillHubScanFinding,
  SkillHubScanResult,
  SkillHubSearchResponse,
  SkillHubSource,
  SkillHubSourcesResponse,
  SkillInfo,
  StaleAuxAssignment,
  StarmapCluster,
  StarmapEdge,
  StarmapGraph,
  StarmapMemoryCard,
  StarmapNode,
  StatusResponse,
  TerminalBackendInfo,
  TerminalBackendStatus,
  TerminalBackendsResponse,
  ToolEnvVar,
  ToolProvider,
  ToolProviderStatus,
  ToolsetConfig,
  ToolsetInfo,
  ToolsetModel,
  ToolsetModelsResponse,
  UsageStats,
  WebCapability,
} from '@hermes/shared/gateway-contracts'

export type * from '@hermes/shared/gateway-contracts'

export type OAuthStartResponse =
  | {
      auth_url: string
      expires_in: number
      flow: 'pkce'
      session_id: string
    }
  | {
      expires_in: number
      flow: 'device_code'
      poll_interval: number
      session_id: string
      user_code: string
      verification_url: string
    }

export interface CustomEndpointsResponse {
  current: {
    base_url: string
    model: string
    provider: string
  }
  endpoints: CustomEndpoint[]
  id?: string
  ok?: boolean
}

export interface HermesConfig {
  agent?: {
    reasoning_effort?: string
    personalities?: Record<string, unknown>
    service_tier?: string
  }
  display?: {
    personality?: string
    skin?: string
    interim_assistant_messages?: boolean
  }
  desktop?: {
    repo_scan_enabled?: boolean
    repo_scan_roots?: string[]
    repo_scan_exclude_paths?: string[]
  }
  terminal?: {
    cwd?: string
  }
  stt?: {
    enabled?: boolean
  }
  voice?: {
    max_recording_seconds?: number
    auto_tts?: boolean
  }
}

export type HermesConfigRecord = Record<string, unknown>

export interface PaginatedSessions {
  limit: number
  offset: number
  sessions: SessionInfo[]
  total: number
  /** Listable conversation count per profile (children excluded), keyed by
   *  profile name. Lets the sidebar scope its "Load more" footer to the active
   *  profile instead of the global total. Present only on
   *  `/api/profiles/sessions`. */
  profile_totals?: Record<string, number>
  /** Per-profile read failures from the cross-profile aggregator (e.g. a locked
   *  or corrupt state.db). Present only on `/api/profiles/sessions`. */
  errors?: Array<{ profile: string; error: string }>
}

export interface RpcEvent<T = unknown> {
  payload?: T
  profile?: string
  session_id?: string
  type: string
}

export type TimelineDisplayMetadata =
  | { model: string; provider?: string }
  | { delegation_id: string; task_count: number; completed_count?: number; failed_count?: number; duration_seconds?: number }

export interface SessionResumeResponse {
  inflight?: null | {
    assistant?: string
    streaming?: boolean
    user?: string
  }
  queued?: null | {
    user?: string
  }
  info?: SessionRuntimeInfo
  message_count: number
  messages: SessionMessage[]
  resumed: string
  running?: boolean
  session_id: string
  session_key?: string
  started_at?: number
  status?: string
}

export type SessionRuntimeInfo = GatewaySessionRuntimeInfo

export interface AnalyticsResponse {
  by_model: AnalyticsModelEntry[]
  daily: AnalyticsDailyEntry[]
  period_days: number
  skills: {
    summary: AnalyticsSkillsSummary
    top_skills: AnalyticsSkillEntry[]
  }
  /** Per-tool-name call counts. Absent on older backends. */
  tools?: AnalyticsToolEntry[]
  totals: AnalyticsTotals
}

export interface AuxiliaryModelsResponse {
  main: { model: string; provider: string }
  tasks: AuxiliaryTaskAssignment[]
}

export interface MoaConfigResponse {
  default_preset: string
  active_preset: string
  presets: Record<
    string,
    {
      aggregator: MoaModelSlot
      aggregator_temperature: number
      enabled: boolean
      max_tokens: number
      reference_models: MoaModelSlot[]
      reference_temperature: number
      /** Optional advisor output cap — round-tripped, not edited here. */
      reference_max_tokens?: number | null
      /** Fan-out cadence (per_iteration | user_turn) — round-tripped. */
      fanout?: string
    }
  >
  aggregator: MoaModelSlot
  aggregator_temperature: number
  enabled: boolean
  max_tokens: number
  reference_models: MoaModelSlot[]
  reference_temperature: number
}

export interface McpServerTestResponse {
  ok: boolean
  error?: string
  tools: { name: string; description: string }[]
}

/** One Nous-approved MCP catalog entry from `GET /api/mcp/catalog`. */
export interface McpCatalogEntry {
  name: string
  description: string
  source: string
  transport: string
  auth_type: string
  required_env: { name: string; prompt: string; required: boolean }[]
  command: string | null
  args: string[]
  url: string | null
  install_url: string | null
  install_ref: string | null
  bootstrap: string[]
  default_enabled: string[] | null
  post_install: string
  needs_install: boolean
  installed: boolean
  enabled: boolean
}

export interface McpCatalogResponse {
  entries: McpCatalogEntry[]
  diagnostics: { name: string; kind: string; message: string }[]
}

/** `GET /api/memory` — active provider + built-in memory file sizes. */
export interface MemoryStatusResponse {
  active: string
  providers: { name: string; description: string; configured: boolean }[]
  builtin_files: { memory: number; user: number }
}
