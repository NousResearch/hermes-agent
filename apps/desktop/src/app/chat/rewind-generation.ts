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
 *
 * True = the page outranks the rendered transcript (an intentional rewind
 * landed between the two reads) — the caller must NOT retain the rendered
 * rows the page omits. A page with no generation (older backend) or a
 * rendered transcript with none yet (first paint) never outranks: the guard
 * only ever disables retention, never enables dropping rows the graft would
 * otherwise keep.
 */
export function pageOutranksRenderedTranscript(
  page: Pick<SessionMessagesResponse, 'rewind_generation'> | null | undefined,
  rendered: Pick<ClientSessionState, 'rewindGeneration'> | undefined
): boolean {
  const pageGeneration = page?.rewind_generation
  const renderedGeneration = rendered?.rewindGeneration

  return (
    typeof pageGeneration === 'number' &&
    typeof renderedGeneration === 'number' &&
    pageGeneration > renderedGeneration
  )
}
