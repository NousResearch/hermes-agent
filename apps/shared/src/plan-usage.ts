/**
 * Usage-window plan chrome shared by every model picker (desktop dialog + composer menu, TUI).
 *
 * The gateway sends each model's word per caller (`usage_level` on `/v1/models`); the backend copies
 * it into `ModelOptionProvider.usage_levels`. Non-window providers have no `usage_levels` / `plan_usage`, so every helper
 * returns the empty answer and their pickers are unchanged.
 *
 * Words are each client's own translated copy (desktop `t.modelPicker`, TUI `pickers.model.plan`).
 * Python twin: hermes_cli/model_picker_usage.py.
 */
import type { ModelOptionProvider, PlanUsageGauge, UsageLimit } from './gateway-contract.generated.js'

/** A usage-window plan's limit on a refused turn (error_surface.usage_limit, agent/plan_limits.py). */
export type { UsageLimit }

export type UsageLevel = 'free' | 'low' | 'medium' | 'top_tier'

const USAGE_LEVELS = new Set<string>(['free', 'low', 'medium', 'top_tier'])

type PlanProvider = Partial<Pick<ModelOptionProvider, 'plan_usage' | 'usage_levels'>> | null | undefined

export function usageLevel(provider: PlanProvider, model: string): UsageLevel | null {
  // Window plans only: any other account keeps its main-branch row even if the field were sent.
  if (provider?.plan_usage?.kind !== 'usage_window') {
    return null
  }

  const level = provider.usage_levels?.[model]

  return level && USAGE_LEVELS.has(level) ? (level as UsageLevel) : null
}

/** The tag a picker row shows: only Large (it has its own monthly limit) and Free (never runs out). Small /
 *  Medium change only how fast usage drains, so they carry no tag. A row billed to a subscription the account
 *  holds (`plan_usage.subscription_model_ids`) keeps its `free` level, so it is never dimmed, but carries no
 *  tag: "Free" would read as free of charge when that subscription pays. */
export function usageTag(provider: PlanProvider, model: string): 'free' | 'top_tier' | null {
  const level = usageLevel(provider, model)

  if (level && (provider?.plan_usage?.subscription_model_ids ?? []).includes(model)) {
    return null
  }

  return level === 'free' || level === 'top_tier' ? level : null
}

/** True when any listed model is Large (the footnote shows only then). */
export function hasTopTier(provider: PlanProvider, models: readonly string[]): boolean {
  return models.some(model => usageLevel(provider, model) === 'top_tier')
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
 * room. Same rule as agent/plan_limits.py `blocking_gauge`: a free
 * model (gateway word `free`) is never blocked; a full
 * `all` gauge blocks every paid model; a full `top_tier` gauge blocks the gateway's top-tier models. Of
 * several full gauges, the one that frees up last.
 */
export function blockingGauge(provider: PlanProvider | null | undefined, model: string): PlanUsageGauge | undefined {
  const usage = provider?.plan_usage

  if (usage?.kind !== 'usage_window') {
    return undefined
  }

  if (usageLevel(provider, model) === 'free' || model.endsWith(':free')) {
    return undefined
  }

  const topTier = usageLevel(provider, model) === 'top_tier' || (usage.top_tier_model_ids ?? []).includes(model)

  const full = (usage.gauges ?? []).filter(
    g => g.left_fraction <= 0 && (gaugeScope(g) === 'all' || (gaugeScope(g) === 'top_tier' && topTier))
  )

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
