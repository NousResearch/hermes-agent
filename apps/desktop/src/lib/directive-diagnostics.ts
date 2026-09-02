/**
 * DIRECTIVE DIAGNOSTICS — make a dropped widget say so.
 *
 * A transcript directive fails silently by construction. When the parse or the
 * claim misses, the paragraph renders as its own raw source (`::followup{…}`),
 * which reads like the MODEL emitted junk rather than like the APP dropped a
 * widget — so nobody files it as a bug, and finding the cause needs a bisect.
 *
 * These warnings are for the console/desktop.log only; the transcript itself
 * keeps degrading gracefully to text. Each distinct problem is logged ONCE per
 * session: a streaming message re-renders its paragraphs on every token, so an
 * un-deduped warning would emit hundreds of lines per message.
 */

import {
  describeDirectiveParseFailure,
  looksLikeDirective,
  parseTranscriptDirective
} from '@/lib/transcript-directives'

/** Warnings already emitted, keyed by their dedupe key. */
const seen = new Set<string>()

/** Bound the set on adversarial/long sessions — diagnostics must never leak. */
const MAX_SEEN = 500

function warnOnce(key: string, ...args: unknown[]): void {
  if (seen.has(key)) {
    return
  }

  if (seen.size >= MAX_SEEN) {
    seen.clear()
  }

  seen.add(key)
  console.warn(...args)
}

/** Test seam: diagnostics dedupe across a session, tests need a clean slate. */
export function resetDirectiveDiagnostics(): void {
  seen.clear()
}

/**
 * A paragraph that ADDRESSES a directive but does not parse.
 *
 * Skipped while the message is streaming: a half-arrived directive is
 * malformed by definition (`::followup{p1="Bắn m`), and warning on it would
 * fire on nearly every token of every directive ever emitted.
 */
export function warnUnparsedDirective(text: string, streaming: boolean): void {
  if (streaming || !looksLikeDirective(text)) {
    return
  }

  const reason = describeDirectiveParseFailure(text)

  if (!reason) {
    return
  }

  const trimmed = text.trim()

  warnOnce(
    `parse:${trimmed.slice(0, 120)}`,
    `[transcript-directive] ignored a directive-looking paragraph: ${reason}`,
    trimmed.length > 240 ? `${trimmed.slice(0, 240)}…` : trimmed
  )
}

/**
 * A directive that PARSED but no plugin claims.
 *
 * The common cause is a plugin that failed to load or was disabled — the model
 * kept emitting the directive it was told about while the UI silently stopped
 * rendering it. Names the registered alternatives so a typo is obvious.
 */
export function warnUnclaimedDirective(name: string, registered: readonly string[], streaming: boolean): void {
  if (streaming) {
    return
  }

  warnOnce(
    `unclaimed:${name}`,
    `[transcript-directive] no plugin claims "::${name}" — rendering it as text.`,
    registered.length > 0 ? `Registered: ${registered.join(', ')}` : 'No directive plugins are registered.'
  )
}

/**
 * A claimed directive whose plugin render threw.
 *
 * The contribution error boundary already catches this and shows an inline
 * chip, but the chip is easy to miss in a long transcript and says nothing
 * about WHICH directive died. Name it in the log next to the plugin id.
 */
export function warnDirectiveRenderFailed(name: string, contributionId: string, error: unknown): void {
  const message = error instanceof Error ? error.message : String(error)

  warnOnce(
    `render:${contributionId}:${message}`,
    `[transcript-directive] "::${name}" (${contributionId}) failed to render: ${message}`,
    error
  )
}

/**
 * Why a directive-looking paragraph will render as a badge instead of a
 * widget, phrased for a user rather than a maintainer.
 *
 * Returns null when the paragraph is NOT a dropped directive, which is the
 * common case and must stay cheap: ordinary prose, and any directive that
 * renders correctly, take this path on every paragraph of every message.
 *
 * Streaming is never a drop: every prefix of an arriving directive is
 * malformed, so a badge would flicker through the whole emission.
 */
export function describeDirectiveDrop(
  text: string,
  registeredNames: readonly string[],
  streaming: boolean
): string | null {
  if (streaming || !looksLikeDirective(text)) {
    return null
  }

  if (describeDirectiveParseFailure(text)) {
    return 'Malformed panel skipped'
  }

  const parsed = parseTranscriptDirective(text)

  if (!parsed) {
    return null
  }

  // Parsed cleanly and someone claims it: not a drop. The render-failure case
  // is handled by the contribution boundary, which owns its own fallback.
  if (registeredNames.includes(parsed.name)) {
    return null
  }

  return registeredNames.length > 0
    ? `Panel "${parsed.name}" is unavailable`
    : 'Panel plugin not loaded'
}
