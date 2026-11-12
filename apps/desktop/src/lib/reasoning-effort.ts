import { DEFAULT_REASONING_EFFORT, isReasoningEffort, type ReasoningEffort } from '@hermes/shared'

import { runtimeTranslations } from '@/i18n'
import { normalize } from '@/lib/text'

/** Compact labels for chrome where space is tight (pill, picker rows), from the
 *  catalog (`shell.modelOptions.effortShort`). React callers pass their `t`
 *  copy so the label re-renders with the language; other code reads the live
 *  runtime catalog. */
type EffortShortLabels = Readonly<Record<string, string>>

/**
 * A pick the route does not send verbatim: `ultra` is a Hermes-internal step
 * that routes clamp to their strongest level (`max` on OpenAI-compatible wires; the Codex
 * app-server sends it verbatim as codex's own mode), and the
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

/** Compact label; a clamped pick shows both ends ("Ultra→Max") so the pill
 *  never presents a Hermes step as a wire level the route does not have. */
export function reasoningEffortLabel(
  effort: string,
  wire?: string,
  labels: EffortShortLabels = runtimeTranslations().shell.modelOptions.effortShort
): string {
  const key = normalize(effort)
  const clamp = reasoningEffortClamp(effort, wire)

  if (clamp) {
    return `${labels[clamp.effort]}→${labels[clamp.wire]}`
  }

  return key ? (labels[key] ?? effort) : ''
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
