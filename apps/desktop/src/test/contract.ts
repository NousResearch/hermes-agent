/**
 * Builders for the generated wire types. The contract spells every field, so a
 * test states only what it asserts on and inherits the rest at its zero value.
 */
import type {
  AgentPluginRow,
  ApprovalParams,
  ClarifyBatch,
  ClarifySingle,
  CommandsCatalogResult,
  ConnectorRow,
  GoalSnapshot,
  HeartbeatSnapshot,
  InflightTurn,
  LoopSnapshot,
  MessageCompletePayload,
  MessageDeltaPayload,
  MoaReferencePayload,
  ModelOptionProvider,
  ModelOptionsResult,
  PetGalleryEntry,
  PetInfoResult,
  ProcessEntry,
  PromptSubmitResult,
  SessionActiveItem,
  SessionCompressResult,
  SessionControlDispatch,
  SessionControlSnapshot,
  SessionInfoPayload,
  SessionLiveInfo,
  SessionResumeResult,
  SetupRuntimeCheckResult,
  SetupStatusResult,
  TodoItem,
  ToolCompletePayload,
  ToolStartPayload,
  TranscriptMessage,
  Usage,
  WakeStartResult,
  WakeStatusResult,
  WakeStopResult
} from '@hermes/shared'

export const usage = (over: Partial<Usage> = {}): Usage => ({
  model: '',
  input: 0,
  output: 0,
  reasoning: 0,
  prompt: 0,
  completion: 0,
  total: 0,
  calls: 0,
  compressions: null,
  context_used: null,
  context_max: null,
  context_percent: null,
  context_source: null,
  context_estimated: null,
  cache_hit_pct: null,
  cache_read: null,
  cache_write: null,
  avg_latency_s: null,
  avg_tps: null,
  active_subagents: null,
  dev_credits_spent_micros: null,
  cost_usd: null,
  cost_status: null,
  ...over
})

/** The `info` block of resume/activate/create method results. */
export const sessionLiveInfo = (over: Partial<SessionLiveInfo> = {}): SessionLiveInfo => ({
  model: null,
  provider: '',
  reasoning_effort: '',
  service_tier: '',
  fast: false,
  yolo: false,
  approval_mode: '',
  tools: null,
  skills: null,
  cwd: '',
  branch: null,
  project: null,
  terminal_backend: '',
  personality: '',
  running: false,
  turn_started_at: null,
  title: '',
  stored_session_id: '',
  desktop_contract: null,
  version: '',
  release_date: '',
  update_behind: null,
  update_command: '',
  usage: null,
  profile_name: null,
  mcp_servers: [],
  system_prompt: null,
  credential_warning: null,
  lazy: null,
  ...over
})

/** `session.info` event payload: `SessionLiveInfo` plus the event-only `config_warning`. */
export const sessionInfoPayload = (over: Partial<SessionInfoPayload> = {}): SessionInfoPayload => ({
  ...sessionLiveInfo(),
  config_warning: null,
  ...over
})

export const transcriptMessage = (over: Partial<TranscriptMessage> = {}): TranscriptMessage => ({
  role: 'user',
  text: null,
  timestamp: null,
  row_id: null,
  display_kind: null,
  display_metadata: null,
  name: null,
  context: null,
  args: null,
  reasoning: null,
  reasoning_content: null,
  reasoning_details: null,
  codex_reasoning_items: null,
  codex_message_items: null,
  ...over
})

/** `session.resume` and `session.activate` answer the same row. */
export const sessionResumeResult = (over: Partial<SessionResumeResult> = {}): SessionResumeResult => ({
  session_id: '',
  message_count: 0,
  messages: [],
  pending_connection: null,
  info: sessionLiveInfo(),
  stored_session_id: null,
  resumed: null,
  session_key: null,
  messages_omitted: null,
  hydrating: null,
  running: null,
  turn_started_at: null,
  started_at: null,
  status: null,
  inflight: null,
  queued: null,
  pending_approval: null,
  open_requests: null,
  todo_state: null,
  auto_continue: null,
  ...over
})

export const sessionCompressResult = (over: Partial<SessionCompressResult> = {}): SessionCompressResult => ({
  status: null,
  removed: null,
  before_messages: null,
  after_messages: null,
  before_tokens: null,
  after_tokens: null,
  summary: null,
  usage: null,
  info: null,
  messages: null,
  compressed: null,
  lock_held: null,
  message: null,
  turn_isolation: null,
  host_ack: null,
  ...over
})

export const inflightTurn = (over: Partial<InflightTurn> = {}): InflightTurn => ({
  assistant: '',
  streaming: false,
  user: '',
  display_kind: null,
  display_metadata: null,
  corrections: null,
  correction_offsets: null,
  error: null,
  status: null,
  recoverable: null,
  error_surface: null,
  ...over
})

export const promptSubmitResult = (over: Partial<PromptSubmitResult> = {}): PromptSubmitResult => ({
  status: null,
  voice_stopped: null,
  survivor_user_row_ids: null,
  survivor_row_id_map: null,
  turn_isolation: null,
  ...over
})

export const wakeStatusResult = (over: Partial<WakeStatusResult> = {}): WakeStatusResult => ({
  listening: false,
  owned_by_caller: false,
  owner_surface: null,
  phrase: '',
  provider: '',
  configured_surface: '',
  input_device: {
    selector: null,
    name: null,
    error: null,
    max_input_channels: null,
    default_samplerate: null,
    hostapi_index: null,
    hostapi: null
  },
  available: false,
  hint: '',
  enabled: false,
  audio_silent: false,
  capture: '',
  local_input_available: false,
  sample_rate: 0,
  frame_length: 0,
  ...over
})

export const wakeStartResult = (over: Partial<WakeStartResult> = {}): WakeStartResult => ({
  started: false,
  reason: null,
  hint: null,
  phrase: null,
  provider: null,
  owner_surface: null,
  enabled_persisted: null,
  capture: null,
  sample_rate: null,
  frame_length: null,
  ...over
})

export const wakeStopResult = (over: Partial<WakeStopResult> = {}): WakeStopResult => ({
  reason: null,
  stopped: false,
  disabled_persisted: false,
  ...over
})

export const sessionActiveItem = (over: Partial<SessionActiveItem> = {}): SessionActiveItem => ({
  current: false,
  id: '',
  last_active: 0,
  message_count: 0,
  model: '',
  preview: '',
  session_key: '',
  started_at: 0,
  status: 'idle',
  title: '',
  ...over
})

export const agentPluginRow = (over: Partial<AgentPluginRow> = {}): AgentPluginRow => ({
  name: '',
  key: '',
  version: '',
  description: '',
  source: '',
  status: '',
  portable: false,
  install_dir: '',
  has_desktop_half: false,
  catalog_name: null,
  catalog_tier: null,
  installed_sha: null,
  catalog_sha: null,
  catalog_version: null,
  update_available: null,
  pinned_sha: null,
  ...over
})

export const modelOptionProvider = (over: Partial<ModelOptionProvider> = {}): ModelOptionProvider => ({
  slug: '',
  name: '',
  models: [],
  total_models: 0,
  is_current: false,
  is_user_defined: false,
  source: '',
  aliases: null,
  api_url: null,
  native_catalog_empty: null,
  auth_type: null,
  authenticated: null,
  key_env: null,
  warning: null,
  featured_models: null,
  capabilities: null,
  pricing: null,
  pricing_pending: null,
  free_tier: null,
  free_tier_pending: null,
  free_tier_row: null,
  unavailable_models: null,
  ...over
})

export const modelOptionsResult = (over: Partial<ModelOptionsResult> = {}): ModelOptionsResult => ({
  providers: [],
  model: '',
  provider: '',
  ...over
})

export const petInfoResult = (over: Partial<PetInfoResult> = {}): PetInfoResult => ({
  enabled: false,
  slug: null,
  displayName: null,
  mime: null,
  spritesheetBase64: null,
  spritesheetRevision: null,
  spritesheetUnchanged: null,
  frameW: null,
  frameH: null,
  framesPerState: null,
  framesByState: null,
  framesByRow: null,
  loopMs: null,
  scale: null,
  stateRows: null,
  ...over
})

export const petGalleryEntry = (over: Partial<PetGalleryEntry> = {}): PetGalleryEntry => ({
  slug: '',
  displayName: '',
  installed: false,
  spritesheetUrl: '',
  curated: null,
  generated: false,
  ...over
})

export const setupStatusResult = (over: Partial<SetupStatusResult> = {}): SetupStatusResult => ({
  provider_configured: null,
  ready: null,
  free_tier: null,
  other_providers: null,
  inference_provider: null,
  profile: null,
  error_code: null,
  retryable: null,
  retry_after: null,
  ok: null,
  error: null,
  ...over
})

export const setupRuntimeCheckResult = (over: Partial<SetupRuntimeCheckResult> = {}): SetupRuntimeCheckResult => ({
  ok: false,
  provider: null,
  model: null,
  source: null,
  error: null,
  free_tier: null,
  profile: null,
  ...over
})

export const processEntry = (over: Partial<ProcessEntry> = {}): ProcessEntry => ({
  session_id: '',
  command: '',
  cwd: null,
  pid: null,
  owner_task_id: null,
  started_at: '',
  uptime_seconds: 0,
  status: 'running',
  output_preview: '',
  output_tail: '',
  session_scoped: null,
  watch_patterns: null,
  watch_hit: null,
  notify_on_complete: null,
  exit_code: null,
  detached: null,
  ...over
})

export const commandsCatalogResult = (over: Partial<CommandsCatalogResult> = {}): CommandsCatalogResult => ({
  pairs: [],
  sub: {},
  canon: {},
  commands: {},
  categories: [],
  skills: {},
  skill_count: 0,
  warning: '',
  ...over
})

export const connectorRow = (over: Partial<ConnectorRow> = {}): ConnectorRow => ({
  connector: '',
  connected: false,
  enabled: false,
  connectionStatus: null,
  name: null,
  description: null,
  ...over
})

export const clarifySingle = (over: Partial<ClarifySingle> = {}): ClarifySingle => ({
  session_id: '',
  kind: 'single',
  question: '',
  choices: null,
  multi_select: false,
  ...over
})

export const clarifyBatch = (over: Partial<ClarifyBatch> = {}): ClarifyBatch => ({
  session_id: '',
  kind: 'batch',
  questions: [],
  answers: null,
  ...over
})

export const approvalParams = (over: Partial<ApprovalParams> = {}): ApprovalParams => ({
  session_id: '',
  request_id: '',
  command: '',
  description: '',
  pattern_key: null,
  pattern_keys: null,
  allow_permanent: null,
  allow_session: null,
  smart_denied: null,
  choices: ['once', 'session', 'always', 'deny'],
  ...over
})

export const messageDeltaPayload = (over: Partial<MessageDeltaPayload> = {}): MessageDeltaPayload => ({
  text: '',
  rendered: null,
  verbose: null,
  ...over
})

export const messageCompletePayload = (over: Partial<MessageCompletePayload> = {}): MessageCompletePayload => ({
  text: '',
  usage: null,
  status: null,
  reasoning: null,
  warning: null,
  response_previewed: null,
  billing: null,
  failure_reason: null,
  rendered: null,
  error: null,
  recoverable: null,
  error_surface: null,
  partial: null,
  ...over
})

export const toolStartPayload = (over: Partial<ToolStartPayload> = {}): ToolStartPayload => ({
  tool_id: '',
  name: '',
  context: null,
  args: null,
  args_text: null,
  preview: null,
  ...over
})

export const toolCompletePayload = (over: Partial<ToolCompletePayload> = {}): ToolCompletePayload => ({
  tool_id: '',
  name: '',
  args: null,
  duration_s: null,
  result: null,
  summary: null,
  result_text: null,
  inline_diff: null,
  todos: null,
  revision: null,
  ...over
})

export const moaReferencePayload = (over: Partial<MoaReferencePayload> = {}): MoaReferencePayload => ({
  label: '',
  text: '',
  index: null,
  count: null,
  ...over
})

export const todoItem = (over: Partial<TodoItem> = {}): TodoItem => ({
  id: '',
  content: '',
  status: 'pending',
  parent: null,
  ...over
})

export const goalSnapshot = (over: Partial<GoalSnapshot> = {}): GoalSnapshot => ({
  title: '',
  status: 'active',
  turns_used: 0,
  max_turns: 0,
  contract: { outcome: '', verification: '', constraints: '', boundaries: '', stop_when: '' },
  subgoals: [],
  gates: [],
  created_at: null,
  updated_at: null,
  paused_reason: null,
  last_verdict: null,
  last_reason: null,
  wait_barrier: null,
  ...over
})

export const loopSnapshot = (over: Partial<LoopSnapshot> = {}): LoopSnapshot => ({
  prompt: '',
  status: 'active',
  mode: 'interval',
  interval_seconds: 0,
  current_delay: 0,
  times: 0,
  until: '',
  max_ticks: 0,
  ticks_fired: 0,
  created_at: 0,
  last_fired_at: 0,
  next_due_at: 0,
  awaiting_response: false,
  deferred_by_goal: false,
  paused_reason: null,
  last_stop_reason: null,
  ...over
})

export const heartbeatSnapshot = (over: Partial<HeartbeatSnapshot> = {}): HeartbeatSnapshot => ({
  prompt: '',
  status: 'active',
  interval_seconds: 0,
  created_at: 0,
  last_fired_at: 0,
  fire_count: 0,
  ...over
})

export const sessionControlSnapshot = (over: Partial<SessionControlSnapshot> = {}): SessionControlSnapshot => ({
  goal: null,
  loop: null,
  heartbeat: null,
  revision: '',
  updated_at: 0,
  ...over
})

export const sessionControlDispatch = (over: Partial<SessionControlDispatch> = {}): SessionControlDispatch => ({
  type: null,
  output: null,
  notice: null,
  message: null,
  display: null,
  ...over
})
