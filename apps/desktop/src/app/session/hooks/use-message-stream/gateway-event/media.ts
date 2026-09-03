import { recordMediaDeliverable } from '@/lib/media-store'

import type { GatewayEventContext } from './types'

/**
 * `media.deliverable` (D1): the gateway extracted a real, validated media file
 * from a finished reply. Record its metadata so the renderer's media cards can
 * label with the true kind/mime/size instead of guessing from the extension,
 * and so a ref whose file later becomes unreachable still renders a *named*
 * fallback card.
 *
 * Routing is free: the frame carries `session_id` and rides the standard
 * session event transport (seq-stamped + replayed on reconnect like every
 * other event), so `ctx.sessionId` has already been resolved when we get here.
 * Recording is session-agnostic by design — the same file re-tagged in another
 * turn refreshes the row.
 *
 * Ordering fact this relies on (serve path, tui_gateway/server.py): the media
 * frames are emitted BEFORE `message.complete`, so the size/kind are in the
 * registry when the final text's `MEDIA:` tags are turned into links.
 *
 * Zero-silent: this handler never drops a payload silently. A malformed
 * payload is rejected by `recordMediaDeliverable` (returns false, row not
 * registered) and the associated ref then renders through the fallback card
 * with an unknown-metadata shape — visible, never blank.
 */
export function handleMediaEvent(ctx: GatewayEventContext): boolean {
  if (ctx.event.type !== 'media.deliverable') {
    return false
  }

  if (ctx.payload) {
    recordMediaDeliverable(ctx.payload, ctx.occurredAt)
  }

  return true
}
