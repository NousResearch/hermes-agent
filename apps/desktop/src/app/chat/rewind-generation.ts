import type { ClientSessionState } from '@/app/types'
import type { SessionMessagesResponse } from '@/types/hermes'

/**
 * Rewind-generation gate for stale-page retention (#119819).
 *
 * A refreshed page that omits the rendered transcript's newest durable rows is
 * ambiguous on its own:
 *
 * - Page generation EQUALS the rendered transcript's — the read is STALE (a
 *   slow/partial REST response racing a live tile): the rendered rows are real
 *   and must be retained.
 * - Page generation GREATER than the rendered transcript's — an /undo, /retry
 *   or truncation happened in between; the missing rows are gone on purpose
 *   and the page is the authority. Retaining them would resurrect undone work
 *   on every background refresh (the /undo regression class).
 * - NEITHER side carries a generation (an older backend, or a transcript
 *   painted before any page carried one) — a below-water page is then
 *   indistinguishable from an intentional rewind, so the refresh stays
 *   authoritative exactly as it was before this guard: retention is a
 *   same-generation judgement and requires positive evidence on both sides.
 *     The accepted trade-off: on backends that predate `rewind_generation`
 *     the original #119819 symptom (a stale page clobbering newer rendered
 *     rows) stays possible — without a generation there is no way to tell a
 *     stale read from a rewind, and the pre-guard behavior is the safer
 *     default because it never resurrects undone rows.
 *
 * True = the page outranks the rendered transcript (an intentional rewind
 * landed between the two reads, or no generation evidence exists) — the
 * caller must NOT retain the rendered rows the page omits.
 */
export function pageOutranksRenderedTranscript(
  page: Pick<SessionMessagesResponse, 'rewind_generation'> | null | undefined,
  rendered: Pick<ClientSessionState, 'rewindGeneration'> | undefined
): boolean {
  const pageGeneration = page?.rewind_generation
  const renderedGeneration = rendered?.rewindGeneration

  if (typeof pageGeneration !== 'number' || typeof renderedGeneration !== 'number') {
    return true
  }

  return pageGeneration > renderedGeneration
}

/**
 * The rewind generation a painted transcript was read at (#119819): stamped on
 * the session-state publish so the next stale-page guard can tell a stale read
 * from an intentional rewind. Empty when the page carried no generation or the
 * page never painted.
 */
export function rewindGenerationStamp(
  generation: number | undefined,
  painted: boolean
): { rewindGeneration: number } | Record<string, never> {
  return typeof generation === 'number' && painted ? { rewindGeneration: generation } : {}
}
