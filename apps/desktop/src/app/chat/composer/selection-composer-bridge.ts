/**
 * Selection-to-composer bridge module.
 *
 * Shared primitives for getting selections from any surface into the composer.
 * This module carries the message-quote facility: quoted text from a chat
 * message, stored keyed by messageId and inserted as `@message:<messageId>`
 * refs. The refs freeze into ```quote blocks at the transport boundary (see
 * `freezeComposerTransportPayload` in `@/store/composer`) alongside
 * `@terminal:` chips. The `@message:` reference kind is registered in
 * `reference-kinds.ts` and rendered by `directive-text.tsx`.
 *
 * The message-quote storage lives in `@/store/composer` (alongside
 * `$composerTerminalSelections`) so the transport freeze can resolve it
 * alongside terminal selections. This module is the insertion/adapter layer
 * that surfaces those primitives to UI components.
 */

import { requestComposerInsert } from '@/app/chat/composer/focus'
import { formatRefValue } from '@/components/assistant-ui/directive-text'
import {
  $composerMessageQuotes,
  clearComposerMessageQuotes,
  messageQuoteContextBlocks,
  reconcileComposerMessageQuotes,
  setComposerMessageQuote
} from '@/store/composer'

// Re-export the message-quote store + helpers so callers only need to import
// from this module. The storage itself lives in composer.ts (the established
// home for draft-scoped selection state), matching $composerTerminalSelections.
export {
  $composerMessageQuotes,
  clearComposerMessageQuotes,
  messageQuoteContextBlocks,
  reconcileComposerMessageQuotes,
  setComposerMessageQuote
}

/**
 * Insert a "quote this message" ref into the composer.
 *
 * Stores the quoted text under the messageId (via {@link setComposerMessageQuote})
 * and inserts the `@message:<messageId>` ref text inline so the user can see
 * and remove it before sending. The ref carries the same id the store is
 * keyed by, so the frozen transport resolves it back to the quoted text —
 * a display label in the ref would break that lookup.
 *
 * @param text - The quoted message text.
 * @param messageId - Stable identifier of the source message.
 */
export function addMessageSelectionToChat(text: string, messageId: string): void {
  const trimmed = text.trim()
  const normalizedId = messageId.trim()

  if (!trimmed || !normalizedId) {
    return
  }

  setComposerMessageQuote(normalizedId, trimmed)

  const refText = `@message:${formatRefValue(normalizedId)}`

  requestComposerInsert(refText, { mode: 'inline' })
}
