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
  // The chat's working folder was moved or deleted: nothing ran, and nothing will until it is back
  // or the chat moves to another folder.
  'workspace_unavailable',
  // The API key this chat started with changed or was removed (gateway: a chat never switches keys
  // on its own): Retry repeats it; a new chat runs on the current key.
  'launch_credentials_unavailable',
  // The owner did not confirm a forwarded send in time; it may already be queued: no Retry.
  'SESSION_OWNER_UNCONFIRMED',
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

/** "HH:mm (in 1h 05m)" for a provider reset moment, or null once it has passed
 *  (a Retry then simply works, so the hint disappears). `now` is injectable for tests. */
export function formatLimitReset(resetsAt: number | undefined, now: number = Date.now()): null | string {
  if (typeof resetsAt !== 'number' || !Number.isFinite(resetsAt)) {
    return null
  }

  const remainingMinutes = Math.ceil((resetsAt * 1000 - now) / 60_000)

  if (remainingMinutes <= 0) {
    return null
  }

  const hours = Math.floor(remainingMinutes / 60)
  const minutes = remainingMinutes % 60
  const wait = hours > 0 ? `${hours}h ${String(minutes).padStart(2, '0')}m` : `${minutes}m`

  return `${formatResetClock(resetsAt)} (in ${wait})`
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

  return (ERROR_CODE_KEYS as readonly string[]).includes(surface.code)
    ? { code: surface.code as ErrorCodeKey }
    : { layer: surface.layer }
}

/** Which recovery buttons the error card offers for a failure. Every flag
 *  maps to an EXISTING handler in the app; the card only decides visibility. */
export interface ErrorRecoveryPlan {
  /** assistant-ui reload of the failed turn. */
  retry: boolean
  /** Open the model picker (model_not_found). */
  chooseModel: boolean
  /** Run /compress on this session (context too long). */
  compress: boolean
  /** requestFreshSession — when this session cannot continue as-is. */
  startNewSession: boolean
  /** Open the preceding user message in the edit composer (safety refusal). */
  editMessage: boolean
  /** Reveal the Hermes data folder so the user can free space (disk_full). */
  openHermesFolder: boolean
  /** Settings → Keys deep link (auth, api_key). */
  updateApiKey: boolean
  /** Re-run the provider's OAuth sign-in (auth, oauth). */
  signInAgain: boolean
  /** Open the free-tier sign-in dialog (free_tier_* codes): signing in is free and lifts the refusal. */
  signInFreeTier: boolean
  /** Open the live session model menu (switches THIS session via
   *  model.switch); Settings → Models deep link fallback when no chat surface
   *  is on screen. */
  switchProvider: boolean
  /** Settings → Billing (usage_window: turn on Keep going, add credits). */
  openBilling: boolean
  /** usage_window: one-click "Switch to <recommended model>" (this session),
   *  then re-run the turn. Shown only when the backend named a model. */
  switchToModel: boolean
  /** usage_window: "Use credits" — turns on Keep going, then re-runs the turn. */
  useCredits: boolean
  /** Open logs / Send diagnostics / Copy details. Off for a plan limit: it is
   *  not a fault, and those buttons would bury the two that fix it. */
  diagnostics: boolean
}

// Layers where the fix is provider/endpoint/auth config, not a retry.
const SWITCH_PROVIDER_LAYERS: readonly ErrorSurfaceLayer[] = ['auth', 'billing', 'endpoint', 'provider']

// Per-code overrides on top of the layer defaults. `retry: false` here means
// the classifier may call the failure retryable at the transport level, but
// an unchanged retry is known to reproduce it for the user (too long a
// conversation, a blocked prompt, a chat another surface owns).
const CODE_PLANS: Partial<Record<ErrorCodeKey, Partial<ErrorRecoveryPlan>>> = {
  SESSION_NOT_OWNED: { retry: false, startNewSession: true },
  SESSION_OWNER_BUSY: { retry: true, startNewSession: false },
  SESSION_OWNER_UNCONFIRMED: { retry: false, startNewSession: false },
  cron_session_read_only: { retry: false, startNewSession: true },
  // Retry is the next click once the folder is back or the chat was moved (Move to project).
  workspace_unavailable: { retry: true, startNewSession: true },
  launch_credentials_unavailable: { retry: false, startNewSession: true },
  content_policy_blocked: { editMessage: true, retry: false },
  context_overflow: { compress: true, retry: false, startNewSession: true },
  disk_full: { openHermesFolder: true, retry: true },
  loop_error: { startNewSession: true },
  model_not_found: { chooseModel: true, retry: false },
  payload_too_large: { compress: true, retry: false, startNewSession: true },
  // A window plan's limit: the two ways on, one click each — switch to the
  // plan's recommended model, or spend credits (Use credits when Keep going is
  // off and there is a balance, else Add credits via Billing). Retry at the
  // reset stays. Switch provider is not the fix (the provider is fine).
  usage_window: { diagnostics: false, openBilling: true, retry: true, switchProvider: false }
}

export function errorRecoveryPlan(surface: ErrorSurface | null | undefined): ErrorRecoveryPlan {
  const oauthReauth = isOAuthReauthSurface(surface)
  const apiKeyRejected = isApiKeyRejectedSurface(surface)

  const base: ErrorRecoveryPlan = {
    chooseModel: false,
    compress: false,
    editMessage: false,
    openHermesFolder: false,
    // Retry re-runs the failed prompt in place. Suppressed when the classifier
    // says the failure is deterministic — except for a credential rejection,
    // where fixing the credential changes the outcome and Retry is the
    // natural second click.
    retry: !surface || surface.retryable || oauthReauth || apiKeyRejected,
    signInAgain: oauthReauth,
    signInFreeTier: isFreeTierSurface(surface),
    startNewSession: false,
    switchProvider: surface != null && SWITCH_PROVIDER_LAYERS.includes(surface.layer),
    openBilling: false,
    switchToModel: false,
    useCredits: false,
    diagnostics: true,
    updateApiKey: apiKeyRejected
  }

  const key = errorCardKey(surface)
  const plan = { ...base, ...('code' in key ? CODE_PLANS[key.code] : undefined) }

  if (surface?.code === 'usage_window') {
    const limit = surface.usageLimit
    plan.switchToModel = Boolean(limit?.suggestions.length)
    plan.useCredits = limit?.credits === 'use_credits'
    // Use credits replaces Open billing; Add credits is Open billing.
    plan.openBilling = !plan.useCredits
    // An older backend (no structured limit): keep the menu so there is still a way on.
    plan.switchProvider = !limit
  }

  return plan
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
