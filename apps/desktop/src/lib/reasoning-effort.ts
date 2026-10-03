import { DEFAULT_REASONING_EFFORT, isReasoningEffort, type ReasoningEffort } from '@hermes/shared'

import { normalize } from '@/lib/text'

/** Full chip labels for picker rows; menus and settings use the translated
 *  `shell.modelOptions` strings instead. */
const FULL_LABELS: Record<string, string> = {
  none: 'Off',
  minimal: 'Min',
  low: 'Low',
  medium: 'Med',
  high: 'High',
  xhigh: 'XHigh',
  max: 'Max',
  ultra: 'Ultra'
}

/**
 * A pick the route does not send verbatim: `ultra` is a Hermes-internal step
 * that every route clamps to its strongest level (`max` on OpenAI-compatible wires), and the
 * CLI's `/reasoning` says so ("ultra (sends max on this route)"). The wire
 * level comes from the gateway's `session.info.reasoning_effort_wire`; nothing
 * is inferred client-side, so an unknown ('' — not yet stamped, or an
 * optimistic pick) or verbatim wire reads as "no clamp".
 */
export function reasoningEffortClamp(
  effort: string,
  wire: string | undefined
): { effort: ReasoningEffort; wire: ReasoningEffort } | null {
  const picked = normalize(effort)
  const sent = normalize(wire ?? '')

  if (!sent || sent === picked || !isReasoningEffort(picked) || !isReasoningEffort(sent)) {
    return null
  }

  return { effort: picked, wire: sent }
}

/** Label for the given level using `labels`; a clamped pick shows both ends
 *  ("Ultra→Max") so a chip never presents a Hermes step as a wire level the
 *  route does not send. */
function formatWith(labels: Record<string, string>, effort: string, wire?: string): string {
  const key = normalize(effort)
  const clamp = reasoningEffortClamp(effort, wire)

  if (clamp) {
    return `${labels[clamp.effort]}→${labels[clamp.wire]}`
  }

  return key ? (labels[key] ?? effort) : ''
}

/** Full chip label. */
export function reasoningEffortLabel(effort: string, wire?: string): string {
  return formatWith(FULL_LABELS, effort, wire)
}

/** Thinking is on unless a level explicitly says otherwise; an empty value
 *  means "inherit", so it resolves through `fallback` first. */
export const isThinkingEnabled = (effort: string, fallback: string = DEFAULT_REASONING_EFFORT): boolean =>
  normalize(effort || fallback) !== 'none'

/** The level a scale control should show. Empty inherits `fallback`; `none`
 *  (thinking off) selects nothing; anything unrecognized clamps to the default. */
export function resolveReasoningEffort(effort: string, fallback: string = DEFAULT_REASONING_EFFORT): string {
  const value = normalize(effort || fallback)

  if (value === 'none') {
    return ''
  }

  return isReasoningEffort(value) ? value : DEFAULT_REASONING_EFFORT
}
