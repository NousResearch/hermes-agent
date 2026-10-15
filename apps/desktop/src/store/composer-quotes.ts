import { atom } from 'nanostores'

import type { QuoteRole } from '@/lib/quote-message'

/**
 * Pending quote cards above the composer input — the ChatGPT-style quote
 * preview. The quoted text lives here, NOT in the draft: cards render above
 * the input with a remove affordance, and only serialize into markdown
 * blockquote text at submit (see use-composer-submit / use-composer-queue),
 * so the wire format stays exactly what the transcript already renders.
 */
export interface ComposerQuote {
  id: string
  role: QuoteRole
  text: string
  timestamp: string
}

export interface ComposerQuoteScope {
  $quotes: ReturnType<typeof atom<ComposerQuote[]>>
  add(quote: ComposerQuote): void
  clear(): void
  remove(id: string): void
  forSession(key: string | null): ComposerQuoteScope
}

/**
 * Composer quote scopes — one live quote set PER MOUNTED COMPOSER, mirroring
 * the attachment scope (see store/composer.ts): the main chat's scope wraps
 * the module-level atom below; each session tile creates its own so two
 * composers on screen never share parked quotes.
 */
export function createComposerQuoteScope($quotes = atom<ComposerQuote[]>([])): ComposerQuoteScope {
  const sessions = new Map<string, ComposerQuoteScope>()

  return {
    $quotes,
    forSession(key) {
      if (key === null) {
        return this
      }

      let session = sessions.get(key)

      if (!session) {
        session = createComposerQuoteScope()
        sessions.set(key, session)
      }

      return session
    },
    add(quote) {
      $quotes.set([...$quotes.get(), quote])
    },
    clear() {
      if ($quotes.get().length) {
        $quotes.set([])
      }
    },
    remove(id) {
      const current = $quotes.get()
      const next = current.filter(quote => quote.id !== id)

      if (next.length !== current.length) {
        $quotes.set(next)
      }
    }
  }
}

export const mainComposerQuoteScope = createComposerQuoteScope()
