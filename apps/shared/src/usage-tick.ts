import type { PlanGauge, SessionStats, SessionUsagePayload, Usage } from './gateway-contract.generated.js'

/** The owner's `session.usage` tick (`SessionUsagePayload`, canonical counters at the top level or the classic sidecar ticker's nested `usage`) or its `session.stats` reply (`SessionStats`). */
export type UsageTickPayload = SessionStats | SessionUsagePayload

/** What `usageFromTick` computes from top-level counters: the `Usage` fields both status bars read. */
export interface TickUsage {
  active_subagents?: number
  avg_latency_s?: number
  avg_tps?: number
  cache_hit_pct?: number
  calls: number
  completion: number
  compressions?: number
  context_estimated?: boolean
  context_max?: number
  context_percent?: number
  context_source?: string
  context_used?: number
  input: number
  model?: string
  output: number
  plan_gauge?: PlanGauge
  prompt: number
  reasoning: number
  total: number
}

type Count = 'calls' | 'completion' | 'input' | 'output' | 'prompt' | 'reasoning' | 'total'
type CountSource =
  | 'api_calls'
  | 'completion_tokens'
  | 'input_tokens'
  | 'output_tokens'
  | 'prompt_tokens'
  | 'reasoning_tokens'
  | 'total_tokens'

const num = (value: null | number | undefined): number | undefined =>
  value !== null && value !== undefined && Number.isFinite(value) ? value : undefined

const round1 = (value: number) => Math.round(value * 10) / 10
const pct = (part: number, whole: number) => Math.max(0, Math.min(100, Math.round((part / whole) * 100)))

// `TickUsage` key ← owner stats key(s), first non-zero wins (`gateway/session_usage.py::tui_usage`).
const COUNTERS: [Count, CountSource[]][] = [
  ['calls', ['api_calls']],
  ['input', ['input_tokens', 'prompt_tokens']],
  ['output', ['output_tokens', 'completion_tokens']],
  ['reasoning', ['reasoning_tokens']],
  ['prompt', ['prompt_tokens']],
  ['completion', ['completion_tokens']],
  ['total', ['total_tokens']]
]

function contextFields(p: UsageTickPayload): Partial<TickUsage> {
  const used = num(p.last_prompt_tokens) ?? 0
  const max = num(p.context_length) ?? 0
  const compressions = num(p.compression_count)
  // A compressor, sized or not, sets `context_source` (`tui_usage`'s keying; `_get_usage`'s `if comp`).
  const fields: Partial<TickUsage> = p.context_source == null || compressions === undefined ? {} : { compressions }

  if (!used || !max) {
    return fields
  }

  const source = p.context_source || 'provider_usage'
  const clamped = Math.min(used, max)

  return {
    ...fields,
    context_estimated: source !== 'provider_usage',
    context_max: max,
    context_percent: pct(clamped, max),
    context_source: source,
    context_used: clamped
  }
}

function rateFields(p: UsageTickPayload): Partial<TickUsage> {
  const fields: Partial<TickUsage> = {}
  const promptTotal = num(p.prompt_tokens) ?? 0
  const cacheRead = num(p.cache_read_tokens) ?? 0
  const latency = num(p.avg_latency_s)
  const tps = num(p.tokens_per_s)
  const subagents = num(p.active_subagents)

  if (promptTotal > 0 && cacheRead > 0) {
    fields.cache_hit_pct = pct(cacheRead, promptTotal)
  }

  if (latency !== undefined) {
    fields.avg_latency_s = round1(latency)
  }

  if (tps !== undefined) {
    fields.avg_tps = round1(tps)
  }

  if (subagents !== undefined) {
    fields.active_subagents = subagents
  }

  return fields
}

/** A canonical local viewer's `session.usage` tick carries the owner's session-cumulative counters at the top level; the classic sidecar ticker's nested `usage` passes through unchanged. Each figure maps only when its source is present, so a partial tick never zeroes a counter it did not carry (`session_usage.tui_usage` is the Python twin used for `message.complete.usage`). */
export function usageFromTick<Nested = Usage>(
  payload: null | undefined | (UsageTickPayload & { usage?: Nested | null })
): Nested | Partial<TickUsage> | undefined {
  if (!payload) {
    return undefined
  }

  if ('usage' in payload && payload.usage) {
    return payload.usage
  }

  const usage: Partial<TickUsage> = {}

  for (const [key, sources] of COUNTERS) {
    const values = sources.map(source => num(payload[source]))
    const value = values.find(Boolean) ?? values.find(v => v !== undefined)

    if (value !== undefined) {
      usage[key] = value
    }
  }

  const derived: Partial<TickUsage> = { ...usage, ...contextFields(payload), ...rateFields(payload) }

  if (payload.plan_gauge) {
    derived.plan_gauge = payload.plan_gauge
  }

  if (Object.keys(derived).length === 0) {
    return undefined
  }

  return payload.model ? { ...derived, model: payload.model } : derived
}

/** Merge a usage payload into the stored one: a partial tick keeps the fields it omits, but the plan gauge follows the latest complete payload. Both builders (`gateway/session_usage.py::tui_usage`, `tui_gateway/server.py::_get_usage`) always send `calls` and send `plan_gauge` only on a window plan, so a complete payload without it means the session is off the plan (or the allowance reset) and the old gauge goes. */
export function mergeUsage<P extends object, N extends { calls?: number; plan_gauge?: unknown }>(
  previous: P,
  next: N | null | undefined
): P & Partial<N> {
  const merged = { ...previous, ...next }

  if (next?.calls !== undefined && next.plan_gauge == null) {
    delete (merged as { plan_gauge?: unknown }).plan_gauge
  }

  return merged
}
