export type EffortLabelKey = 'minimal' | 'low' | 'medium' | 'high' | 'max'

export const LEGACY_REASONING_EFFORT_OPTIONS = ['minimal', 'low', 'medium', 'high', 'xhigh'] as const
export const LEGACY_REASONING_DEFAULT_OPTIONS = ['none', ...LEGACY_REASONING_EFFORT_OPTIONS] as const

export const EFFORT_LABEL_KEYS: Record<string, EffortLabelKey> = {
  minimal: 'minimal',
  low: 'low',
  medium: 'medium',
  high: 'high',
  xhigh: 'max'
}

const REASONING_ALIASES: Record<string, string> = {
  off: 'none',
  disabled: 'none',
  max: 'xhigh',
  maximum: 'xhigh',
  'extra-high': 'xhigh',
  'extra high': 'xhigh',
  extrahigh: 'xhigh'
}

const CANONICAL_REASONING_EFFORTS = new Set(['none', ...LEGACY_REASONING_EFFORT_OPTIONS])

export function canonicalReasoningEffort(value: unknown): string {
  const raw = String(value ?? '').trim().toLowerCase()

  return REASONING_ALIASES[raw] ?? raw
}

export function defaultReasoningEffort(options: readonly string[]): string {
  return options.includes('medium') ? 'medium' : options.find(option => option !== 'none') ?? options[0] ?? 'medium'
}

export function reasoningEffortOptions(
  reasoningEfforts?: readonly string[],
  fallback: readonly string[] = LEGACY_REASONING_EFFORT_OPTIONS
): string[] {
  const values = reasoningEfforts?.length ? reasoningEfforts : fallback
  const seen = new Set<string>()
  const deduped: string[] = []

  for (const raw of values) {
    const value = canonicalReasoningEffort(raw)

    if (!CANONICAL_REASONING_EFFORTS.has(value) || seen.has(value)) {
      continue
    }

    seen.add(value)
    deduped.push(value)
  }

  return deduped.length ? deduped : [...fallback]
}

export function normalizeReasoningEffort(effort: unknown, options: readonly string[]): string {
  const value = canonicalReasoningEffort(effort || defaultReasoningEffort(options))

  if (value === 'none') {
    return options.includes('none') ? 'none' : ''
  }

  return options.includes(value) ? value : defaultReasoningEffort(options)
}

export function normalizeReasoningEffortForRequest(effort: unknown, options?: readonly string[]): string {
  const normalizedOptions = reasoningEffortOptions(options)
  const value = normalizeReasoningEffort(effort, normalizedOptions)

  return value || defaultReasoningEffort(normalizedOptions)
}

export function reasoningCanDisable(reasoningEfforts?: readonly string[]): boolean {
  return !reasoningEfforts || reasoningEffortOptions(reasoningEfforts).includes('none')
}
