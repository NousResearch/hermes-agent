/**
 * Quote a chat message into the composer.
 *
 * Used by the right-click "Quote" action and the selection pill on user /
 * assistant bubbles. The quote parks as a removable card above the composer
 * input (ChatGPT-style) through the `hermes:composer-quote-add` bus; the
 * draft text is untouched. Only at send time does the card serialize back
 * into markdown blockquote text, so the wire format stays exactly what the
 * transcript already renders.
 *
 * The blockquote uses the existing in-app convention:
 *
 *     > [user 10:32]
 *     > 原文第一行
 *     > 原文第二行
 *
 * so the receiving model can recognize it as a quote when it reads back the
 * transcript. Empty bodies fall back to `[empty message]`; lines that
 * already start with `>` get a leading space so they don't terminate the
 * quote prematurely; trailing whitespace and very long lines are
 * preserved verbatim so the receiver sees the same bytes the sender
 * selected.
 *
 * Surfaces that can only edit text (the in-thread edit composer) or that
 * never answer the acked card add fall back to the original insert-the-block
 * path so a quote is never dropped.
 */
import {
  type ComposerTarget,
  getActiveComposer,
  requestComposerInsert,
  requestComposerQuoteAddAcked
} from '@/app/chat/composer/focus'
import type { ComposerQuote } from '@/store/composer-quotes'

export type QuoteRole = 'assistant' | 'user'

const ROLE_LABEL: Record<QuoteRole, string> = {
  assistant: 'assistant',
  user: 'user'
}

/** Best-effort HH:MM stamp for the current locale. Reads `Date#toLocaleTimeString`
 *  with the short 24-hour form so the header line fits on one row and sorts
 *  cleanly when read back. */
export function formatMessageTimestamp(date: Date = new Date()): string {
  try {
    return new Intl.DateTimeFormat(undefined, {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    }).format(date)
  } catch {
    const pad = (n: number) => String(n).padStart(2, '0')

    return `${pad(date.getHours())}:${pad(date.getMinutes())}`
  }
}

/** Format a message into a markdown blockquote. Exposed for unit tests and
 *  for any UI surface that wants to render the same shape. */
export function formatQuoteBlock(role: QuoteRole, text: string, timestamp: string): string {
  const trimmed = text.replace(/\s+$/u, '')
  const body = trimmed.length > 0 ? trimmed : '[empty message]'
  const headerLine = `> [${ROLE_LABEL[role]} ${timestamp}]`

  // A line beginning with `>` would close the quote and start a new block —
  // prefix every body line with one space so the quote stays open.
  const quotedLines = body
    .split('\n')
    .map(line => (line.startsWith('>') ? `>  ${line}` : `> ${line}`))
    .join('\n')

  return `${headerLine}\n${quotedLines}\n\n`
}

/** Serialize pending quote cards into the outbound message text: one
 *  blockquote per card, above the typed draft. Used at submit/enqueue time so
 *  the cards themselves never live in the draft. */
export function serializeComposerQuotes(quotes: readonly ComposerQuote[]): string {
  return quotes.map(quote => formatQuoteBlock(quote.role, quote.text, quote.timestamp)).join('')
}

/** Quote a message into the active composer as a pending card. Safe to call
 *  from any bubble handler. The target is pinned at call time (the bus defers
 *  delivery); when no card-capable surface claims it — the in-thread edit
 *  composer being the common case — the quote falls back to a direct
 *  blockquote insert so it is never dropped. */
export function quoteMessageIntoComposer(
  role: QuoteRole,
  text: string,
  timestamp: string,
  target: ComposerTarget = getActiveComposer()
): void {
  const body = text.trim() ? text : '[empty message]'

  if (target === 'edit') {
    requestComposerInsert(formatQuoteBlock(role, body, timestamp), { mode: 'block', target })

    return
  }

  void requestComposerQuoteAddAcked({ id: crypto.randomUUID(), role, text: body, timestamp }, { target }).then(
    claimed => {
      if (!claimed) {
        requestComposerInsert(formatQuoteBlock(role, body, timestamp), { mode: 'block', target })
      }
    }
  )
}
