// Structured turn-error descriptor forwarded by the gateway (see
// agent/error_surface.py). Names WHICH layer of the stack failed and the
// classifier's failure code so the error card can say what happened in plain
// words and offer code-appropriate recovery actions, instead of toasting an
// opaque string.
//
// Advisory contract: older backends never send this — every consumer must
// keep working when it is absent (legacy string-sniffing stays as fallback).

export const ERROR_SURFACE_LAYERS = [
  'provider',
  'endpoint',
  'streaming',
  'auth',
  'billing',
  'gateway',
  'runtime',
  'disk'
] as const

export type ErrorSurfaceLayer = (typeof ERROR_SURFACE_LAYERS)[number]

/** Failure codes the error card has dedicated copy for. Values are what the
 *  backend stamps in `failure_reason` (agent/error_classifier.py
 *  FailoverReason) plus the gateway's own site codes (SESSION_NOT_OWNED,
 *  disk_full, stream_drop). Anything else falls back to the layer copy. */
export const ERROR_CODE_KEYS = [
  'auth',
  'auth_permanent',
  'billing',
  'rate_limit',
  'upstream_rate_limit',
  'overloaded',
  'server_error',
  'timeout',
  'stream_drop',
  'ssl_cert_verification',
  // A firewall/CDN in front of the endpoint refused the request (usually by
  // User-Agent) before it reached the model: not a key problem, and not retryable.
  'upstream_blocked',
  'context_overflow',
  'payload_too_large',
  'model_not_found',
  'provider_policy_blocked',
  'content_policy_blocked',
  'format_error',
  'truncated',
  'invalid_response',
  'empty_response',
  'loop_error',
  'SESSION_NOT_OWNED',
  // The gateway that owns the chat is mid-turn and could not queue this send: transient.
  'SESSION_OWNER_BUSY',
  // The chat is a scheduled job's session: it only runs that job, so a typed message never runs there.
  'cron_session_read_only',
  // Stored history the gateway shows but cannot continue (an old cron run, subagent, imported or
  // WebUI chat): prompt.submit refuses it with this reason.
  'session_read_only',
  // The chat's working folder was moved or deleted: nothing ran, and nothing will until it is back
  // or the chat moves to another folder.
  'workspace_unavailable',
  // The API key this chat started with changed or was removed (gateway: a chat never switches keys
  // on its own): Retry repeats it; a new chat runs on the current key.
  'launch_credentials_unavailable',
  // The managed local model server is down (starting, or switched off): named as itself, not as
  // the custom-server endpoint copy, and with no Desktop-only pane (the Webapp shares this card).
  'local_runtime_unavailable',
  // The owner did not confirm a forwarded send in time; it may already be queued: no Retry.
  'SESSION_OWNER_UNCONFIRMED',
  'agent_init_failed',
  'disk_full',
  // Raised by the desktop, never by the backend: the backend reported a turn
  // over after its events stopped, and no reply reached this window.
  'no_reply',
  // The Nous free tier refused or could not serve the turn (agent/error_surface.py
  // `free_tier_<kind>`). The backend's sentence rides in `message` and is the card body.
  'free_tier_disabled',
  'free_tier_rate_limited',
  'free_tier_at_capacity',
  'free_tier_model_not_free',
  'free_tier_route',
  'free_tier_outage',
  'free_tier_refused',
  // A usage-window plan limit ran out and credits won't pay. The backend's
  // stop-line sentence (which limit, the reset, the way on) rides in `message`.
  'usage_window'
] as const

export type ErrorCodeKey = (typeof ERROR_CODE_KEYS)[number]

export interface ErrorSurface {
  layer: ErrorSurfaceLayer
  /** Specific failure code (a FailoverReason value or site-specific code). */
  code: string
  /** False when retrying unchanged reproduces the same failure. */
  retryable: boolean
  /** The failing session's provider/model, captured at classification time —
   *  preferred over the foreground composer's atoms, which can point at a
   *  different model by the time the user clicks an action. */
  provider?: string
  model?: string
  /** Auth layer only: how the failing provider is credentialed. `oauth` means
   *  the fix is signing in again (expired/revoked grant); `api_key` means a
   *  key needs replacing. Absent from older backends. */
  authKind?: 'api_key' | 'oauth'
  /** Auth layer only: display name of the failing provider ("Nous Portal"). */
  providerLabel?: string
  /** Auth layer, api_key only: the env var holding the rejected key
   *  (OPENAI_API_KEY). Deep-links Settings → Keys to that row. Absent from
   *  older backends. */
  apiKeyEnv?: string
  /** Free-tier codes: the backend's own plain sentence for this failure (it
   *  names the wait, the model, the way forward). Shown as the card body. */
  message?: string
  /** Epoch seconds when the provider said its limit lifts (Retry-After header /
   *  `resets_at` body field on a 429). Rendered as "Limit resets at HH:mm" next
   *  to Retry. Absent when the provider named no reset or on older backends. */
  resetsAt?: number
  /** usage_window only: the plan's recommended models that still work past the
   *  limit (best first) and the credits option — the card's one-click buttons
   *  (agent/plan_limits.py). Absent on older backends. */
  usageLimit?: UsageLimitSurface
}

export interface UsageLimitSuggestion {
  model: string
  label: string
  /** 'free' | 'included' */
  note: string
  provider: string
}

export interface UsageLimitSurface {
  suggestions: UsageLimitSuggestion[]
  /** agent/plan_limits.py::credits_action. use_credits: Keep going is off and there is a balance; add_credits:
   * top up; add_credits_and_keep_going: both. Only use_credits gets its own button, the others open billing. */
  credits?: 'add_credits' | 'add_credits_and_keep_going' | 'use_credits'
}

function parseUsageLimit(value: unknown): UsageLimitSurface | undefined {
  if (!value || typeof value !== 'object') {
    return undefined
  }

  const raw = value as { credits?: { action?: unknown } | null; suggestions?: unknown }

  const suggestions = (Array.isArray(raw.suggestions) ? raw.suggestions : []).flatMap(item => {
    const s = item as Partial<Record<keyof UsageLimitSuggestion, unknown>>

    return typeof s?.model === 'string' && s.model
      ? [
          {
            model: s.model,
            label: typeof s.label === 'string' && s.label ? s.label : s.model,
            note: typeof s.note === 'string' ? s.note : '',
            provider: typeof s.provider === 'string' && s.provider ? s.provider : 'nous'
          }
        ]
      : []
  })

  const action = raw.credits?.action

  const credits =
    action === 'use_credits' || action === 'add_credits' || action === 'add_credits_and_keep_going' ? action : undefined

  return suggestions.length || credits ? { suggestions, ...(credits ? { credits } : {}) } : undefined
}

/** Validate a wire payload into an ErrorSurface, or null when absent/garbled. */
// health: allow CC -- usage-window: one branch for the optional usageLimit payload (parsed in parseUsageLimit); over target on main
export function parseErrorSurface(value: unknown): ErrorSurface | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const raw = value as {
    api_key_env?: unknown
    auth_kind?: unknown
    code?: unknown
    layer?: unknown
    message?: unknown
    model?: unknown
    provider?: unknown
    provider_label?: unknown
    resets_at?: unknown
    retryable?: unknown
    usage_limit?: unknown
  }

  const layer = typeof raw.layer === 'string' ? (raw.layer as ErrorSurfaceLayer) : null

  if (!layer || !ERROR_SURFACE_LAYERS.includes(layer)) {
    return null
  }

  const usageLimit = parseUsageLimit(raw.usage_limit)

  return {
    layer,
    code: typeof raw.code === 'string' && raw.code ? raw.code : 'unknown',
    retryable: raw.retryable !== false,
    ...(typeof raw.provider === 'string' && raw.provider ? { provider: raw.provider } : {}),
    ...(typeof raw.model === 'string' && raw.model ? { model: raw.model } : {}),
    ...(raw.auth_kind === 'oauth' || raw.auth_kind === 'api_key' ? { authKind: raw.auth_kind } : {}),
    ...(typeof raw.provider_label === 'string' && raw.provider_label ? { providerLabel: raw.provider_label } : {}),
    ...(typeof raw.api_key_env === 'string' && raw.api_key_env ? { apiKeyEnv: raw.api_key_env } : {}),
    ...(typeof raw.message === 'string' && raw.message.trim() ? { message: raw.message.trim() } : {}),
    ...(typeof raw.resets_at === 'number' && Number.isFinite(raw.resets_at) && raw.resets_at > 0
      ? { resetsAt: raw.resets_at }
      : {}),
    ...(usageLimit ? { usageLimit } : {})
  }
}

/** "HH:mm" for a provider reset moment, in the user's local clock. */
export function formatResetClock(resetsAt: number): string {
  const at = new Date(resetsAt * 1000)

  return `${String(at.getHours()).padStart(2, '0')}:${String(at.getMinutes()).padStart(2, '0')}`
}

/** Browsers clamp `setTimeout` delays to a signed 32-bit millisecond count and
 *  fire anything larger immediately — a reset that far out gets no schedule
 *  button at all rather than an instant (and pointless) retry. */
const MAX_TIMER_DELAY_MS = 2 ** 31 - 1

/** Milliseconds until the card may fire its one scheduled retry, or null when
 *  the reset already passed (Retry works now) or is too far out to time. */
export function scheduledRetryDelayMs(resetsAt: number | undefined, now: number = Date.now()): null | number {
  if (typeof resetsAt !== 'number' || !Number.isFinite(resetsAt)) {
    return null
  }

  const delay = resetsAt * 1000 - now

  return delay > 0 && delay <= MAX_TIMER_DELAY_MS ? delay : null
}

/** "12m 03s" / "1h 05m 03s" for the live countdown on a scheduled retry. */
export function formatCountdown(remainingMs: number): string {
  const totalSeconds = Math.max(0, Math.ceil(remainingMs / 1000))
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = totalSeconds % 60
  const tail = `${String(minutes).padStart(2, '0')}m ${String(seconds).padStart(2, '0')}s`

  return hours > 0 ? `${hours}h ${tail}` : tail.replace(/^0/, '')
}

/** True when the Nous free tier refused or could not serve the turn: the way
 *  forward is the free sign-in (or another provider), never an OAuth re-login. */
export function isFreeTierSurface(surface: ErrorSurface | null | undefined): boolean {
  return typeof surface?.code === 'string' && surface.code.startsWith('free_tier_')
}

/** True when the failed turn's provider rejected an OAuth grant — the
 *  one-click recovery is re-running that provider's sign-in, not editing keys. */
export function isOAuthReauthSurface(surface: ErrorSurface | null | undefined): surface is ErrorSurface & {
  authKind: 'oauth'
  provider: string
} {
  return surface?.layer === 'auth' && surface.authKind === 'oauth' && Boolean(surface.provider)
}

/** True when the failed turn's provider rejected a saved API key — the fix is
 *  replacing the key in Settings → Keys, then retrying. */
export function isApiKeyRejectedSurface(surface: ErrorSurface | null | undefined): surface is ErrorSurface & {
  authKind: 'api_key'
} {
  return surface?.layer === 'auth' && surface.authKind === 'api_key'
}

/** Which copy entry the error card (and the matching toast) should read:
 *  the code table when the code is one we have words for, else the layer
 *  table, else `generic` (no descriptor — older backend). One resolver so the
 *  inline card and the global toast never disagree about the same failure. */
export type ErrorCardKey = { code: ErrorCodeKey } | { layer: 'generic' | ErrorSurfaceLayer }

export function errorCardKey(surface: ErrorSurface | null | undefined): ErrorCardKey {
  if (!surface) {
    return { layer: 'generic' }
  }

  // A custom/local model server gets the endpoint copy ("Can't reach your
  // model server"), not the hosted-provider copy that says to check the internet.
  if (surface.layer === 'endpoint' && (surface.code === 'timeout' || surface.code === 'ssl_cert_verification')) {
    return { layer: 'endpoint' }
  }

  return (ERROR_CODE_KEYS as readonly string[]).includes(surface.code)
    ? { code: surface.code as ErrorCodeKey }
    : { layer: surface.layer }
}

// Layers where another provider can be the fix (the provider side failed, not the gateway or disk).
const SWITCH_PROVIDER_LAYERS: readonly ErrorSurfaceLayer[] = ['auth', 'billing', 'endpoint', 'provider']

/** Open logs / Send diagnostics / Copy details. Off for a plan limit: it is not a fault,
 *  and those links would bury the actions that fix it. */
export function errorCardShowsDiagnostics(surface: ErrorSurface | null | undefined): boolean {
  return surface?.code !== 'usage_window'
}

/** Plain-text error-details blob for the error card's "Copy error details". */
export function formatErrorDiagnostics(input: {
  appVersion?: string
  errorText: string
  /** Epoch seconds the failure was recorded (the bubble's own time), never the copy time. */
  failedAt?: number
  model?: string
  provider?: string
  surface?: ErrorSurface | null
}): string {
  // The descriptor's identity (captured when the turn failed) beats the
  // caller-supplied fallback (typically the foreground composer's atoms).
  const provider = input.surface?.provider || input.provider
  const model = input.surface?.model || input.model

  const lines = [
    '── Hermes error details ──',
    input.failedAt ? `time: ${new Date(input.failedAt * 1000).toISOString()}` : null,
    input.surface ? `layer: ${input.surface.layer}` : null,
    input.surface ? `code: ${input.surface.code}` : null,
    input.surface ? `retryable: ${input.surface.retryable}` : null,
    input.surface?.resetsAt ? `resets_at: ${new Date(input.surface.resetsAt * 1000).toISOString()}` : null,
    provider ? `provider: ${provider}` : null,
    model ? `model: ${model}` : null,
    input.appVersion ? `app: ${input.appVersion}` : null,
    `error: ${input.errorText}`
  ]

  return lines.filter((line): line is string => Boolean(line)).join('\n')
}

// ── Error card v2: one primary action, at most one text link ─────────────
//
// The Error System Overhaul (Plan tab): every failure gets exactly ONE
// primary button that fixes the cause, plus at most one text link for a
// backup fix the body already names. Diagnostics (copy / logs / send) live
// inside the collapsed details block, never in the action row.

export type ErrorActionId =
  | 'retry'
  | 'retry_at_reset'
  | 'update_api_key'
  | 'sign_in_again'
  | 'sign_in_free_tier'
  | 'switch_provider'
  | 'choose_model'
  | 'compress'
  | 'start_new_session'
  | 'edit_message'
  | 'open_hermes_folder'
  | 'continue'
  | 'open_provider_settings'
  | 'open_setup_guide_tls'
  | 'open_setup_guide_local'
  | 'open_model_settings'
  | 'switch_to_model'
  | 'use_credits'
  | 'open_billing'

export interface ErrorCardActions {
  primary: ErrorActionId | null
  secondary?: ErrorActionId
}

export const ERROR_SETUP_GUIDE_URLS = {
  tls: 'https://hermes-agent.nousresearch.com/docs/user-guide/security#tls-certificate-trust',
  local: 'https://hermes-agent.nousresearch.com/docs/user-guide/local-models#using-your-own-llama-server'
} as const

/** Provider pages where the user fixes an account-level routing/privacy block. */
const PROVIDER_SETTINGS_URLS: Record<string, string> = {
  openrouter: 'https://openrouter.ai/settings/privacy',
  nous: 'https://portal.nousresearch.com'
}

/** The provider's own settings page (privacy/data policy), when Hermes knows one. */
export function providerSettingsUrl(surface?: ErrorSurface | null): string | undefined {
  return PROVIDER_SETTINGS_URLS[(surface?.provider || '').toLowerCase()]
}

// One main action + at most one second option per code (Error System Overhaul).
const CODE_ACTIONS: Partial<Record<ErrorCodeKey, ErrorCardActions>> = {
  billing: { primary: 'open_billing', secondary: 'switch_provider' },
  content_policy_blocked: { primary: 'edit_message' },
  context_overflow: { primary: 'compress', secondary: 'start_new_session' },
  disk_full: { primary: 'open_hermes_folder', secondary: 'retry' },
  free_tier_at_capacity: { primary: 'sign_in_free_tier', secondary: 'retry' },
  format_error: { primary: 'switch_provider', secondary: 'start_new_session' },
  free_tier_outage: { primary: 'retry' },
  model_not_found: { primary: 'choose_model' },
  payload_too_large: { primary: 'compress', secondary: 'start_new_session' },
  provider_policy_blocked: { primary: 'open_provider_settings', secondary: 'choose_model' },
  SESSION_NOT_OWNED: { primary: 'start_new_session' },
  ssl_cert_verification: { primary: 'open_setup_guide_tls' },
  truncated: { primary: 'continue' },
  upstream_blocked: { primary: 'switch_provider' },
  agent_init_failed: { primary: 'retry', secondary: 'open_model_settings' },
  SESSION_OWNER_BUSY: { primary: 'retry' },
  // It may already be queued: re-sending could run it twice, so no action.
  SESSION_OWNER_UNCONFIRMED: { primary: null },
  cron_session_read_only: { primary: 'start_new_session' },
  session_read_only: { primary: 'start_new_session' },
  workspace_unavailable: { primary: 'retry', secondary: 'start_new_session' },
  launch_credentials_unavailable: { primary: 'start_new_session' },
  loop_error: { primary: 'retry', secondary: 'start_new_session' }
}

// A window plan's limit: switch to the plan's recommended model, else spend credits
// (Use credits when Keep going is off with a balance, else Add credits via Billing).
// An older backend sends no structured limit: Billing, or another model.
// Once the reset has passed the allowance is back, so a plain Retry replaces the scheduled one.
function usageWindowActions(surface: ErrorSurface): ErrorCardActions {
  const limit = surface.usageLimit
  const credits: ErrorActionId = limit?.credits === 'use_credits' ? 'use_credits' : 'open_billing'

  if (limit?.suggestions.length) {
    return { primary: 'switch_to_model', secondary: credits }
  }

  if (!limit) {
    return { primary: credits, secondary: 'choose_model' }
  }

  if (surface.resetsAt === undefined) {
    return { primary: credits }
  }

  if (surface.resetsAt * 1000 <= Date.now()) {
    return { primary: credits, secondary: 'retry' }
  }

  // Still ahead: Retry at the reset, unless it is too far off for a browser timer (a monthly window).
  return scheduledRetryDelayMs(surface.resetsAt) === null
    ? { primary: credits }
    : { primary: credits, secondary: 'retry_at_reset' }
}

/** Actions for a model server the user runs (custom endpoint or the managed local runtime). */
function endpointActions(surface: ErrorSurface, key: ErrorCardKey): ErrorCardActions | null {
  // The managed local model server is down: another model is the fix the card can offer, and a
  // server that is still starting (retryable) is worth one more try.
  if (surface.code === 'local_runtime_unavailable') {
    return surface.retryable ? { primary: 'retry', secondary: 'switch_provider' } : { primary: 'switch_provider' }
  }

  // errorCardKey routes a custom/local server's timeout or TLS failure to the endpoint copy.
  if ('layer' in key && key.layer === 'endpoint') {
    return surface.code === 'timeout'
      ? { primary: 'retry', secondary: 'open_setup_guide_local' }
      : { primary: 'open_setup_guide_local' }
  }

  return null
}

export function errorCardActions(surface: ErrorSurface | null | undefined): ErrorCardActions {
  if (!surface) {
    return { primary: 'retry' }
  }

  // Fixing the credential changes the outcome, so Retry stays as the follow-up click.
  if (surface.layer === 'auth') {
    return {
      primary: surface.authKind === 'oauth' && surface.provider ? 'sign_in_again' : 'update_api_key',
      secondary: 'retry'
    }
  }

  const key = errorCardKey(surface)
  const endpoint = endpointActions(surface, key)

  if (endpoint) {
    return endpoint
  }

  if (surface.code === 'usage_window') {
    return usageWindowActions(surface)
  }

  const byCode = 'code' in key ? CODE_ACTIONS[key.code] : undefined

  // A provider without a known settings page: choosing another model is the fix left.
  if (surface.code === 'provider_policy_blocked' && !providerSettingsUrl(surface)) {
    return { primary: 'choose_model' }
  }

  if (byCode) {
    return byCode
  }

  // Signing in is free and lifts the refusal; a transient one (rate limit) can also just be retried.
  if (isFreeTierSurface(surface)) {
    return surface.retryable ? { primary: 'sign_in_free_tier', secondary: 'retry' } : { primary: 'sign_in_free_tier' }
  }

  if (
    (surface.code === 'rate_limit' || surface.code === 'upstream_rate_limit') &&
    scheduledRetryDelayMs(surface.resetsAt) !== null
  ) {
    return { primary: 'retry_at_reset', secondary: 'retry' }
  }

  if (surface.retryable) {
    return { primary: 'retry' }
  }

  // Another provider only helps when the provider side failed (not the gateway or the disk).
  return { primary: SWITCH_PROVIDER_LAYERS.includes(surface.layer) ? 'switch_provider' : null }
}

/** Whether the card for this failure offers `action` (as its main action or second option). */
export function errorCardOffers(surface: ErrorSurface | null | undefined, action: ErrorActionId): boolean {
  const { primary, secondary } = errorCardActions(surface)

  return primary === action || secondary === action
}

/**
 * Card severity (Error System Overhaul, Figma "Error handling" frame):
 * - blocking: the user can't keep chatting until they act. Rust tint, rust icon.
 * - degraded: chat still works (a retry, a continue, or a fallback model), something needs
 *   attention. White card, muted icon, dismissible.
 * Severity only changes the icon colour and the tint; the layout and buttons are shared.
 */
export type ErrorCardSeverity = 'blocking' | 'degraded'

const DEGRADED_ACTIONS: ReadonlySet<ErrorActionId> = new Set(['retry', 'retry_at_reset', 'continue'])

export function errorCardSeverity(surface: ErrorSurface | null | undefined): ErrorCardSeverity {
  if (surface?.code === 'free_tier_model_not_free') {
    return 'degraded'
  }

  if (surface?.code === 'agent_init_failed') {
    return 'blocking'
  }

  const { primary } = errorCardActions(surface)

  return primary && DEGRADED_ACTIONS.has(primary) ? 'degraded' : 'blocking'
}
