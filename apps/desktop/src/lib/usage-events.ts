import type { UsageStats } from '@/types/hermes'

function finiteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

export function usageFromTokenUsagePayload(payload: unknown): Partial<UsageStats> | null {
  if (!payload || typeof payload !== 'object') {
    return null
  }

  const p = payload as Record<string, unknown>
  const input = finiteNumber(p.input_tokens)
  const output = finiteNumber(p.output_tokens)
  const total = finiteNumber(p.total_tokens)
  const contextUsed = finiteNumber(p.context_tokens)
  const contextMax = finiteNumber(p.context_length)

  const contextPercent =
    finiteNumber(p.context_pct) ??
    (contextUsed !== undefined && contextMax && contextMax > 0
      ? Math.round((contextUsed / contextMax) * 1000) / 10
      : undefined)

  const usage: Partial<UsageStats> = {}

  if (input !== undefined) {
    usage.input = input
  }

  if (output !== undefined) {
    usage.output = output
  }

  if (total !== undefined) {
    usage.total = total
  }

  if (contextUsed !== undefined) {
    usage.context_used = contextUsed
  }

  if (contextMax !== undefined) {
    usage.context_max = contextMax
  }

  if (contextPercent !== undefined) {
    usage.context_percent = contextPercent
  }

  return Object.keys(usage).length ? usage : null
}
