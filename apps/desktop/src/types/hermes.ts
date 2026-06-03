export interface ConfigFieldSchema {
  category?: string
  description?: string
  options?: unknown[]
  type?: 'boolean' | 'list' | 'number' | 'select' | 'string' | 'text'
}

export interface ConfigSchemaResponse {
  category_order?: string[]
  fields: Record<string, ConfigFieldSchema>
}

export interface AudioTranscriptionResponse {
  ok: boolean
  provider?: string
  transcript: string
}

export interface AudioSpeakResponse {
  ok: boolean
  data_url: string
  mime_type: string
  provider?: string
}

export interface ElevenLabsVoice {
  label: string
  name: string
  voice_id: string
}

export interface ElevenLabsVoicesResponse {
  available: boolean
  voices: ElevenLabsVoice[]
}

export interface OAuthProviderStatus {
  error?: string
  expires_at?: null | string
  has_refresh_token?: boolean
  last_refresh?: null | string
  logged_in: boolean
  source?: null | string
  source_label?: null | string
  token_preview?: null | string
}

export interface OAuthProvider {
  cli_command: string
  docs_url: string
  flow: 'device_code' | 'external' | 'loopback' | 'pkce'
  id: string
  name: string
  status: OAuthProviderStatus
}

export interface OAuthProvidersResponse {
  providers: OAuthProvider[]
}

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
  | {
      auth_url: string
      expires_in: number
      flow: 'loopback'
      session_id: string
    }

export interface OAuthSubmitResponse {
  message?: string
  ok: boolean
  status: 'approved' | 'error'
}

export interface OAuthPollResponse {
  error_message?: null | string
  expires_at?: null | number
  session_id: string
  status: 'approved' | 'denied' | 'error' | 'expired' | 'pending'
}

export interface EnvVarInfo {
  advanced: boolean
  category: string
  description: string
  is_password: boolean
  is_set: boolean
  redacted_value: null | string
  tools: string[]
  url: null | string
}

export interface MessagingEnvVarInfo {
  advanced: boolean
  description: string
  is_password: boolean
  is_set: boolean
  key: string
  prompt: string
  redacted_value: null | string
  required: boolean
  url: null | string
}

export interface MessagingHomeChannel {
  chat_id: string
  name: string
  platform: string
  thread_id?: string
}

export interface MessagingPlatformInfo {
  configured: boolean
  description: string
  docs_url: string
  enabled: boolean
  env_vars: MessagingEnvVarInfo[]
  error_code?: null | string
  error_message?: null | string
  gateway_running: boolean
  home_channel?: MessagingHomeChannel | null
  id: string
  name: string
  state?: null | string
  updated_at?: null | string
}

export interface MessagingPlatformsResponse {
  platforms: MessagingPlatformInfo[]
}

export interface MessagingPlatformUpdate {
  clear_env?: string[]
  enabled?: boolean
  env?: Record<string, string>
}

export interface MessagingPlatformTestResponse {
  message: string
  ok: boolean
  state?: null | string
}

export interface GatewayReadyPayload {
  skin?: unknown
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
  }
  terminal?: {
    cwd?: string
  }
  stt?: {
    enabled?: boolean
  }
  voice?: {
    max_recording_seconds?: number
  }
}

export type HermesConfigRecord = Record<string, unknown>

export interface ModelInfoResponse {
  auto_context_length?: number
  capabilities?: Record<string, unknown>
  config_context_length?: number
  effective_context_length?: number
  model: string
  provider: string
}

export interface ModelPricing {
  /** Formatted $/Mtok input price, e.g. "$3.00", or "free", or "" if unknown. */
  input: string
  /** Formatted $/Mtok output price. */
  output: string
  /** Formatted $/Mtok cached-input price, or null when the model has none. */
  cache: string | null
  /** True when the model costs nothing (free tier eligible). */
  free: boolean
}

export interface ModelOptionProvider {
  is_current?: boolean
  models?: string[]
  name: string
  slug: string
  total_models?: number
  warning?: string
  /** Per-model pricing keyed by model id (present when the picker requested
   *  pricing and the provider supports live pricing). */
  pricing?: Record<string, ModelPricing>
  /** Nous only: whether the current account is on the free tier. */
  free_tier?: boolean
  /** Nous only: paid models a free-tier user cannot select (shown disabled). */
  unavailable_models?: string[]
  /** Per-model option support, keyed by model id (present when the picker
   *  requested capabilities). Lets the UI gate fast/reasoning controls. */
  capabilities?: Record<string, ModelCapabilities>
}

export interface ModelCapabilities {
  fast: boolean
  reasoning: boolean
}

export interface ModelOptionsResponse {
  model?: string
  provider?: string
  providers?: ModelOptionProvider[]
}

export interface PaginatedSessions {
  limit: number
  offset: number
  sessions: SessionInfo[]
  total: number
}

export interface RpcEvent<T = unknown> {
  payload?: T
  session_id?: string
  type: string
}

export interface SessionCreateResponse {
  info?: SessionRuntimeInfo
  message_count?: number
  messages?: SessionMessage[]
  session_id: string
  stored_session_id?: string
}

export interface SessionInfo {
  archived?: boolean
  cwd?: null | string
  ended_at: null | number
  id: string
  /** Original root id of a compression chain, when this entry is a projected
   *  continuation tip. Stable across compressions — used as the durable id for
   *  pins so a pinned conversation survives auto-compression. */
  _lineage_root_id?: null | string
  input_tokens: number
  is_active: boolean
  last_active: number
  message_count: number
  model: null | string
  output_tokens: number
  preview: null | string
  source: null | string
  started_at: number
  title: null | string
  tool_call_count: number
}

export interface SessionMessage {
  codex_reasoning_items?: unknown
  content: unknown
  context?: unknown
  name?: string
  reasoning?: null | string
  reasoning_content?: null | string
  reasoning_details?: unknown
  role: 'assistant' | 'system' | 'tool' | 'user'
  text?: unknown
  timestamp?: number
  tool_call_id?: null | string
  tool_calls?: unknown
  tool_name?: string
}

export interface SessionMessagesResponse {
  messages: SessionMessage[]
  session_id: string
}

export interface SessionResumeResponse {
  info?: SessionRuntimeInfo
  message_count: number
  messages: SessionMessage[]
  resumed: string
  session_id: string
}

export interface SessionRuntimeInfo {
  branch?: string
  config_warning?: string
  credential_warning?: string
  cwd?: string
  desktop_contract?: number
  fast?: boolean
  model?: string
  personality?: string
  provider?: string
  reasoning_effort?: string
  running?: boolean
  service_tier?: string
  skills?: Record<string, string[]> | string[]
  tools?: Record<string, string[]>
  usage?: Partial<UsageStats>
  version?: string
}

export interface UsageStats {
  calls: number
  context_max?: number
  context_percent?: number
  context_used?: number
  cost_usd?: number
  input: number
  output: number
  total: number
}

export interface AnalyticsDailyEntry {
  actual_cost: number
  api_calls: number
  cache_read_tokens: number
  day: string
  estimated_cost: number
  input_tokens: number
  output_tokens: number
  reasoning_tokens: number
  sessions: number
}

export interface AnalyticsModelEntry {
  api_calls: number
  estimated_cost: number
  input_tokens: number
  model: string
  output_tokens: number
  sessions: number
}

export interface AnalyticsResponse {
  by_model: AnalyticsModelEntry[]
  daily: AnalyticsDailyEntry[]
  period_days: number
  skills: {
    summary: AnalyticsSkillsSummary
    top_skills: AnalyticsSkillEntry[]
  }
  totals: AnalyticsTotals
}

export interface AnalyticsSkillEntry {
  last_used_at: null | number
  manage_count: number
  percentage: number
  skill: string
  total_count: number
  view_count: number
}

export interface AnalyticsSkillsSummary {
  distinct_skills_used: number
  total_skill_actions: number
  total_skill_edits: number
  total_skill_loads: number
}

export interface AnalyticsTotals {
  total_actual_cost: number
  total_api_calls: null | number
  total_cache_read: null | number
  total_estimated_cost: number
  total_input: null | number
  total_output: null | number
  total_reasoning: null | number
  total_sessions: number
}

export interface CronJob {
  deliver?: null | string
  enabled: boolean
  id: string
  last_error?: null | string
  last_run_at?: null | string
  name?: null | string
  next_run_at?: null | string
  prompt?: null | string
  schedule?: CronJobSchedule
  schedule_display?: null | string
  script?: null | string
  state?: null | string
}

export interface CronJobCreatePayload {
  deliver?: string
  name?: string
  prompt: string
  schedule: string
}

export interface CronJobSchedule {
  display?: string
  expr?: string
  kind?: string
}

export interface CronJobUpdates {
  deliver?: string
  enabled?: boolean
  name?: string
  prompt?: string
  schedule?: string
}

export interface ProfileCreatePayload {
  clone_from_default?: boolean
  name: string
  no_skills?: boolean
}

export interface ProfileInfo {
  has_env: boolean
  is_default: boolean
  model: null | string
  name: string
  path: string
  provider: null | string
  skill_count: number
}

export interface ProfileSetupCommand {
  command: string
}

export interface ProfileSoul {
  content: string
  exists: boolean
}

export interface ProfilesResponse {
  profiles: ProfileInfo[]
}

export interface SkillInfo {
  category: string
  description: string
  enabled: boolean
  name: string
}

export interface ToolsetInfo {
  configured: boolean
  description: string
  enabled: boolean
  label: string
  name: string
  tools: string[]
}

export interface ToolEnvVar {
  key: string
  prompt: string
  url: string | null
  default: string | null
  is_set: boolean
}

export interface ToolProvider {
  name: string
  badge: string
  tag: string
  env_vars: ToolEnvVar[]
  post_setup: string | null
  requires_nous_auth: boolean
  /** True when this is the provider currently written to config (mirrors the
   *  CLI `hermes tools` active-provider detection). */
  is_active: boolean
}

export interface ToolsetConfig {
  name: string
  has_category: boolean
  providers: ToolProvider[]
  /** Name of the currently active provider, or null if none is configured. */
  active_provider: string | null
}

export interface SessionSearchResult {
  /** Lineage root of the matched conversation. Stable across compression and
   *  used as the durable pin id; falls back to session_id when absent. */
  lineage_root?: string | null
  model: string | null
  role: string | null
  /** Live compression tip of the matched conversation — resume by this id. */
  session_id: string
  session_started: number | null
  snippet: string
  source: string | null
}

export interface SessionSearchResponse {
  results: SessionSearchResult[]
}

export interface LogsResponse {
  file: string
  lines: string[]
}

export interface PlatformStatus {
  error_code?: string
  error_message?: string
  state: string
  updated_at: string
}

export interface StatusResponse {
  active_sessions: number
  config_path: string
  config_version: number
  env_path: string
  gateway_exit_reason: string | null
  gateway_health_url: string | null
  gateway_pid: number | null
  gateway_platforms: Record<string, PlatformStatus>
  gateway_running: boolean
  gateway_state: string | null
  gateway_updated_at: string | null
  hermes_home: string
  latest_config_version: number
  release_date: string
  version: string
}

export interface ActionResponse {
  name: string
  ok: boolean
  pid: number
}

export interface ActionStatusResponse {
  exit_code: number | null
  lines: string[]
  name: string
  pid: number | null
  running: boolean
}

export interface AuxiliaryTaskAssignment {
  base_url: string
  model: string
  provider: string
  task: string
}

export interface AuxiliaryModelsResponse {
  main: { model: string; provider: string }
  tasks: AuxiliaryTaskAssignment[]
}

export interface ModelAssignmentRequest {
  model: string
  provider: string
  scope: 'main' | 'auxiliary'
  task?: string
}

export interface ModelAssignmentResponse {
  /** Toolset keys auto-routed through the Nous Tool Gateway as a result of
   *  switching the main provider to Nous. Empty unless provider === 'nous'
   *  and the user is a paid subscriber with unconfigured tools. */
  gateway_tools?: string[]
  model?: string
  ok: boolean
  provider?: string
  reset?: boolean
  scope?: string
  tasks?: string[]
}
