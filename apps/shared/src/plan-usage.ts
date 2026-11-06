/**
 * Usage-window plan chrome shared by every model picker (desktop dialog + composer menu, TUI).
 *
 * The gateway sends each priced model's cost multiplier against one base model per window caller
 * (`usage_multiplier` on each `/v1/models` row, already rounded, and the base model beside `data`); the backend
 * copies them into `ModelOptionProvider.usage_multipliers` / `usage_basis` + `usage_basis_name`, with the models
 * that draw nothing in
 * `usage_free_models`. Non-window providers have none of these nor `plan_usage`, so every helper returns the
 * empty answer and their pickers are unchanged. Clients only format the value (plus a fixed colour by the value, `usageTone`): never derive a
 * multiplier or a tier from prices.
 *
 * Words are each client's own translated copy (desktop `t.modelPicker`, TUI `pickers.model.plan`).
 * Python twin: hermes_cli/model_picker_usage.py.
 */
import type { ModelOptionProvider, PlanUsageGauge, UsageLimit } from './gateway-contract.generated.js'

/** A usage-window plan's limit on a refused turn (error_surface.usage_limit, agent/plan_limits.py). */
export type { UsageLimit }

type PlanProvider =
  | Partial<
      Pick<
        ModelOptionProvider,
        'plan_usage' | 'usage_basis' | 'usage_basis_name' | 'usage_free_models' | 'usage_multipliers'
      >
    >
  | null
  | undefined

/** What a window-plan picker row shows in its chip slot: Free, or the gateway's cost multiplier. */
export type UsageChip = { kind: 'free' } | { kind: 'multiplier'; value: number }

const isWindow = (provider: PlanProvider) => provider?.plan_usage?.kind === 'usage_window'

/** True when *model* draws nothing from the window (a `:free` id or a free catalog row): a full allowance never
 *  dims it. Subscription-billed rows count too (that subscription pays). Window plans only. */
export function isPlanFree(provider: PlanProvider, model: string): boolean {
  return isWindow(provider) && (model.endsWith(':free') || (provider?.usage_free_models ?? []).includes(model))
}

/** A multiplier above this is not a real price ratio: the row shows no chip (Python twin
 *  `MAX_USAGE_MULTIPLIER`). */
export const MAX_USAGE_MULTIPLIER = 100_000

/** The gateway's multiplier for *model* (a finite number above 0, at most {@link MAX_USAGE_MULTIPLIER}), or null.
 *  Window plans only. */
function multiplierOf(provider: PlanProvider, model: string): null | number {
  const value = provider?.usage_multipliers?.[model]

  return typeof value === 'number' && Number.isFinite(value) && value > 0 && value <= MAX_USAGE_MULTIPLIER
    ? value
    : null
}

/**
 * The chip a window-plan picker row shows: `free` (wins over any multiplier) or the gateway's `multiplier`;
 * null for a row with neither and for every other account (even if a field were sent). A row billed to a
 * subscription the account holds (`plan_usage.subscription_model_ids`) is never dimmed but carries no chip:
 * "Free" would read as free of charge when that subscription pays, and the gateway sends it no multiplier.
 */
export function usageChip(provider: PlanProvider, model: string): UsageChip | null {
  if (!isWindow(provider) || (provider?.plan_usage?.subscription_model_ids ?? []).includes(model)) {
    return null
  }

  if (isPlanFree(provider, model)) {
    return { kind: 'free' }
  }

  const value = multiplierOf(provider, model)

  return value === null ? null : { kind: 'multiplier', value }
}

/** `<0.1×` below 0.1, else the gateway's rounded value as sent (`0.4×`, `3×`, `15×`). Python twin:
 *  hermes_cli/model_picker_usage.py `format_usage_multiplier`. */
export function formatUsageMultiplier(value: number): string {
  if (value < 0.1) {
    return '<0.1×'
  }

  // One decimal at most (the gateway sends one below 1), so a float artefact never prints 0.30000000000000004.
  return `${Number.isInteger(value) ? value : Math.round(value * 10) / 10}×`
}

/** How loud a multiplier chip is: `low` (plain), `mid` (amber), `high` (red). Fixed cutoffs on the multiplier
 *  as sent, so the same model reads the same on every account; retune them if the base model changes. */
export type UsageTone = 'high' | 'low' | 'mid'

export const USAGE_TONE_MID_FROM = 15
export const USAGE_TONE_HIGH_FROM = 50

export function usageTone(value: number): UsageTone {
  return value >= USAGE_TONE_HIGH_FROM ? 'high' : value >= USAGE_TONE_MID_FROM ? 'mid' : 'low'
}

/** The base model every multiplier is against, with the gateway's name for it. */
export type UsageBasis = { id: string; name: string }

/** The gateway's base model for this window provider (named by the gateway, else its id), or null. */
export function providerUsageBasis(provider: PlanProvider): null | UsageBasis {
  const id = isWindow(provider) ? provider?.usage_basis : null

  return id ? { id, name: provider?.usage_basis_name || id } : null
}

/**
 * The base model every multiplier is against (`1×`), for the muted `1× = <name> price` line under a window-plan
 * picker's section headings: when the gateway sent one and at least one of *models* shows a multiplier (the
 * base row itself need not be listed). null otherwise and for every other account.
 */
export function usageBasis(provider: PlanProvider, models: readonly string[]): null | UsageBasis {
  const basis = providerUsageBasis(provider)

  return basis && models.some(model => usageChip(provider, model)?.kind === 'multiplier') ? basis : null
}

/**
 * Window plans: `{ recommended, other }` — the plan's featured picks that are listed (plan order),
 * then every other listed model in its given order. `null` for every other account, and when the
 * plan has no featured pick in the list (nothing to split).
 */
export function splitRecommended(
  provider: PlanProvider,
  models: readonly string[]
): { other: string[]; recommended: string[] } | null {
  if (provider?.plan_usage?.kind !== 'usage_window') {
    return null
  }

  const listed = new Set(models)
  const recommended = [...new Set(provider.plan_usage.featured_model_ids ?? [])].filter(model => listed.has(model))

  if (recommended.length === 0) {
    return null
  }

  const head = new Set(recommended)

  return { other: models.filter(model => !head.has(model)), recommended }
}

/** The section title a window-plan picker shows above row *idx* of *models* (no query): `recommended`
 *  above the first featured pick, `other` above the first row after them, else null. *models* must be in
 *  `splitRecommended` order (picks first): a caller that pins an extra row re-orders after the pin. */
export function sectionAt(
  provider: PlanProvider,
  models: readonly string[],
  idx: number
): 'other' | 'recommended' | null {
  const split = splitRecommended(provider, models)

  if (!split) {
    return null
  }

  if (idx === 0) {
    return 'recommended'
  }

  return idx === split.recommended.length && split.other.length > 0 ? 'other' : null
}

/**
 * Server allowance ids with client words (translated per surface). The server sends the list and its
 * order; any other id renders with the gauge's own `title`, so a new allowance needs no client release.
 */
export type KnownGaugeId = 'day' | 'month' | 'top_tier' | 'week'

/** The pause reason that means the subscription payment failed (agent/usage_gauges.py
 *  `PAYMENT_PAUSE_REASONS`). Any other reason, `unknown` included, is a pause with neutral copy. */
export const PAYMENT_PAUSE_REASON = 'payment_past_due'

const KNOWN_GAUGE_IDS = new Set<string>(['day', 'month', 'top_tier', 'week'])

export function knownGaugeId(gauge: Pick<PlanUsageGauge, 'label'>): KnownGaugeId | null {
  return KNOWN_GAUGE_IDS.has(gauge.label) ? (gauge.label as KnownGaugeId) : null
}

/** Which models a full gauge stops: `all` = every paid model, `top_tier` = top-tier models only. */
export function gaugeScope(gauge: Pick<PlanUsageGauge, 'scope'>): 'all' | 'top_tier' {
  return gauge.scope === 'top_tier' ? 'top_tier' : 'all'
}

/** The display name: the client word for a known id, else the server's title (else the id). */
export function gaugeTitle(
  gauge: Pick<PlanUsageGauge, 'label' | 'title'>,
  words: Record<KnownGaugeId, string>
): string {
  const known = knownGaugeId(gauge)

  return known ? words[known] : gauge.title || gauge.label
}

/**
 * The full allowance that stops *model* on a window plan, or undefined while it still has included
 * room. Same rule as agent/plan_limits.py `blocking_gauge`: a free model ({@link isPlanFree}) is never
 * blocked; a full `all` gauge blocks every paid model; a full `top_tier` gauge blocks no row. Of several full
 * gauges, the one that frees up last.
 */
export function blockingGauge(provider: PlanProvider | null | undefined, model: string): PlanUsageGauge | undefined {
  const usage = provider?.plan_usage

  if (usage?.kind !== 'usage_window') {
    return undefined
  }

  if (isPlanFree(provider, model) || model.endsWith(':free')) {
    return undefined
  }

  const full = (usage.gauges ?? []).filter(g => g.left_fraction <= 0 && gaugeScope(g) === 'all')

  return full.reduce<PlanUsageGauge | undefined>(
    (last, g) => (!last || (g.resets_at_ms ?? 0) > (last.resets_at_ms ?? 0) ? g : last),
    undefined
  )
}

/** "5h", "2d 3h", "12m", "soon" until *resetsAtMs* — the picker's compact reset (Python twin:
 *  hermes_cli/model_picker_usage.py `format_resets_in`, without the "resets in" prefix). */
export function formatResetsIn(resetsAtMs: null | number | undefined, nowMs: number = Date.now()): string {
  if (typeof resetsAtMs !== 'number' || resetsAtMs <= 0) {
    return ''
  }

  const minutes = Math.max(0, Math.floor((resetsAtMs - nowMs) / 60_000))

  if (minutes < 1) {
    return 'soon'
  }

  if (minutes < 60) {
    return `${minutes}m`
  }

  const hours = Math.floor(minutes / 60)

  if (hours < 24) {
    return `${hours}h`
  }

  const rest = hours % 24

  return rest ? `${Math.floor(hours / 24)}d ${rest}h` : `${Math.floor(hours / 24)}d`
}

/**
 * True when a window-plan picker row should look unavailable: a full allowance stops it and top-up
 * credits won't pay (pay as you go keeps it usable). The limit line above the list says why, once.
 */
export function planBlocks(provider: PlanProvider | null | undefined, model: string): boolean {
  return provider?.plan_usage?.credits_pay !== true && blockingGauge(provider, model) !== undefined
}

/**
 * The full allowance worth one line at the top of a window-plan picker, and what happens next. An
 * all-model allowance wins over the large-model one (it stops more). `then`: `credits` = paid models (or
 * Large models) now bill top-up credits, pay as you go; `wait` = they stop until the reset; `unknown` =
 * an older portal that does not say. `null` while nothing is full or the plan is paused.
 */
export function planLimit(
  provider: PlanProvider | null | undefined
): { gauge: PlanUsageGauge; scope: 'all' | 'top_tier'; then: 'credits' | 'unknown' | 'wait' } | null {
  const usage = provider?.plan_usage

  if (usage?.kind !== 'usage_window' || usage.paused_reason) {
    return null
  }

  const full = (usage.gauges ?? []).filter(g => g.left_fraction <= 0)

  const latest = (scope: 'all' | 'top_tier') =>
    full
      .filter(g => gaugeScope(g) === scope)
      .reduce<PlanUsageGauge | undefined>(
        (last, g) => (!last || (g.resets_at_ms ?? 0) > (last.resets_at_ms ?? 0) ? g : last),
        undefined
      )

  const all = latest('all')
  const gauge = all ?? latest('top_tier')

  if (!gauge) {
    return null
  }

  const then = usage.credits_pay === true ? 'credits' : usage.credits_pay === false ? 'wait' : 'unknown'

  return { gauge, scope: all ? 'all' : 'top_tier', then }
}

/** The words a terminal picker's limit line is built from; English by default, the TUI passes its
 *  translated copy (desktop renders its own from {@link planLimit}). */
export interface PlanLimitWords {
  resetsIn: (duration: string) => string
  /** "Weekly limit reached" from a limit name ("weekly"). */
  limitReached: (name: string) => string
  limitNames: Record<KnownGaugeId, string>
  limitThenCredits: string
  limitThenCreditsLarge: string
  limitThenWait: string
  limitThenWaitLarge: string
}

export const PLAN_LIMIT_WORDS_EN: PlanLimitWords = {
  resetsIn: duration => `resets in ${duration}`,
  limitReached: name => `${name.charAt(0).toUpperCase()}${name.slice(1)} limit reached`,
  // "You've hit your <name> limit" words (agent/usage_gauges.py `gauge_limit_name`).
  limitNames: { day: 'daily', month: 'monthly', top_tier: 'monthly large-model', week: 'weekly' },
  limitThenCredits: 'paid models now use your credits (pay as you go)',
  limitThenCreditsLarge: 'Large models now use your credits (pay as you go)',
  limitThenWait: 'only Free models until then',
  limitThenWaitLarge: 'Large models paused until then'
}

/** One-line limit summary for terminal pickers:
 *  "Weekly limit reached · resets in 21h · paid models now use your credits (pay as you go)". */
export function planLimitLine(
  provider: PlanProvider | null | undefined,
  nowMs: number = Date.now(),
  words: PlanLimitWords = PLAN_LIMIT_WORDS_EN
): string {
  const limit = planLimit(provider)

  if (!limit) {
    return ''
  }

  const name = gaugeTitle(limit.gauge, words.limitNames)
  const when = formatResetsIn(limit.gauge.resets_at_ms, nowMs)
  // Under a minute: no "resets in soon"; the limit line alone reads right until the next refresh.
  const head = `${words.limitReached(name)}${when && when !== 'soon' ? ` · ${words.resetsIn(when)}` : ''}`
  const all = limit.scope === 'all'

  if (limit.then === 'credits') {
    return `${head} · ${all ? words.limitThenCredits : words.limitThenCreditsLarge}`
  }

  if (limit.then === 'wait') {
    return `${head} · ${all ? words.limitThenWait : words.limitThenWaitLarge}`
  }

  return head
}
