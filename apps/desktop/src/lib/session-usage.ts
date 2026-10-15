import type { UsageStats } from '@/types/hermes'

const number = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined

const percent = (part: number, whole: number) => Math.max(0, Math.min(100, Math.round((part / whole) * 100)))

/** Canonical gateway ticks carry owner counters at the top level; older sidecars
 * wrap display-ready fields in `usage`. Project at the event boundary, matching
 * gateway/session_usage.py::tui_usage, before caching either session's readout. */
export function usageFromTick(payload: unknown): Partial<UsageStats> | undefined {
  if (!payload || typeof payload !== 'object') {
    return undefined
  }

  const p = payload as Record<string, unknown>

  if (p.usage && typeof p.usage === 'object') {
    return p.usage as Partial<UsageStats>
  }

  if (number(p.api_calls) === undefined && number(p.total_tokens) === undefined) {
    return undefined
  }

  const usage: Partial<UsageStats> = {
    calls: number(p.api_calls) ?? 0,
    input: number(p.input_tokens) || number(p.prompt_tokens) || 0,
    output: number(p.output_tokens) || number(p.completion_tokens) || 0,
    total: number(p.total_tokens) ?? 0
  }

  const prompt = number(p.prompt_tokens) ?? 0
  const cached = number(p.cache_read_tokens) ?? 0

  if (prompt > 0 && cached > 0) {
    usage.cache_hit_pct = percent(cached, prompt)
  }

  const tps = number(p.tokens_per_s)

  if (tps !== undefined) {
    usage.avg_tps = Math.round(tps * 10) / 10
  }

  const used = number(p.last_prompt_tokens) ?? 0
  const max = number(p.context_length) ?? 0

  if (used > 0 && max > 0) {
    const source = typeof p.context_source === 'string' && p.context_source ? p.context_source : 'provider_usage'
    usage.context_used = Math.min(used, max)
    usage.context_max = max
    usage.context_percent = percent(usage.context_used, max)
    usage.context_source = source
    usage.context_estimated = source !== 'provider_usage'
  }

  if (p.context_length != null) {
    usage.compressions = number(p.compression_count) ?? 0
  }

  return usage
}
