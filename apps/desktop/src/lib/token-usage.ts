import type { UsageStats } from '@/types/hermes'

export interface TokenUsagePayload {
  context_length?: unknown
  context_pct?: unknown
  context_tokens?: unknown
  input_tokens?: unknown
  output_tokens?: unknown
  total_tokens?: unknown
}

function finiteNumber(value: unknown): number | undefined {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : undefined
  }

  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)

    return Number.isFinite(parsed) ? parsed : undefined
  }

  return undefined
}

function nonNegativeNumber(value: unknown): number | undefined {
  const parsed = finiteNumber(value)

  return parsed === undefined ? undefined : Math.max(0, parsed)
}

function positiveNumber(value: unknown): number | undefined {
  const parsed = finiteNumber(value)

  return parsed === undefined || parsed <= 0 ? undefined : parsed
}

function percentFrom(payload: TokenUsagePayload): number | undefined {
  const explicit = nonNegativeNumber(payload.context_pct)

  if (explicit !== undefined) {
    return Math.min(100, explicit)
  }

  const used = nonNegativeNumber(payload.context_tokens)
  const max = positiveNumber(payload.context_length)

  return used !== undefined && max !== undefined ? Math.min(100, (used / max) * 100) : undefined
}

export function usageFromTokenUsagePayload(payload: TokenUsagePayload | null | undefined): Partial<UsageStats> | null {
  if (!payload) {
    return null
  }

  const usage: Partial<UsageStats> = {}
  const input = nonNegativeNumber(payload.input_tokens)
  const output = nonNegativeNumber(payload.output_tokens)
  const total = nonNegativeNumber(payload.total_tokens)
  const contextUsed = nonNegativeNumber(payload.context_tokens)
  const contextMax = positiveNumber(payload.context_length)
  const contextPercent = percentFrom(payload)

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

export function mergeUsageSnapshot(
  current: UsageStats,
  incoming: Partial<UsageStats> | null | undefined,
  options: { allowContextDecrease?: boolean } = {}
): UsageStats {
  if (!incoming || Object.keys(incoming).length === 0) {
    return current
  }

  const next = { ...current, ...incoming }

  if (options.allowContextDecrease) {
    return next
  }

  const currentContextUsed = finiteNumber(current.context_used)
  const incomingContextUsed = finiteNumber(incoming.context_used)
  const currentContextPercent = finiteNumber(current.context_percent)
  const incomingContextPercent = finiteNumber(incoming.context_percent)
  const incomingCompressionCount = finiteNumber(incoming.compressions)
  const currentCompressionCount = finiteNumber(current.compressions) ?? 0
  const compressionAdvanced =
    incomingCompressionCount !== undefined && incomingCompressionCount > currentCompressionCount
  const contextWindowChanged =
    incoming.context_max !== undefined &&
    current.context_max !== undefined &&
    incoming.context_max !== current.context_max
  const contextWentBackwards =
    (currentContextUsed !== undefined &&
      incomingContextUsed !== undefined &&
      incomingContextUsed < currentContextUsed) ||
    (currentContextPercent !== undefined &&
      incomingContextPercent !== undefined &&
      incomingContextPercent < currentContextPercent)

  if (contextWentBackwards && !compressionAdvanced && !contextWindowChanged) {
    next.context_used = current.context_used
    next.context_percent = current.context_percent
    next.context_max = incoming.context_max ?? current.context_max
  }

  return next
}

export function mergeTokenUsagePayload(
  current: UsageStats,
  payload: TokenUsagePayload | null | undefined
): UsageStats {
  const usage = usageFromTokenUsagePayload(payload)

  return mergeUsageSnapshot(current, usage)
}
