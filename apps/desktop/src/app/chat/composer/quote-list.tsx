import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import type { ComposerQuote } from '@/store/composer-quotes'

/**
 * Pending quote cards above the composer input — the ChatGPT-style quote
 * preview: the quote glyph, one truncated line of the quoted text, and a
 * remove button. The full text and its role/timestamp header stay on the
 * card object and only materialize as a markdown blockquote at submit.
 */
export function QuoteList({ quotes, onRemove }: { quotes: ComposerQuote[]; onRemove?: (id: string) => void }) {
  const { t } = useI18n()

  return (
    <div className="flex max-w-full flex-col gap-1.5 px-1 pt-1" data-slot="composer-quotes">
      {quotes.filter(Boolean).map(quote => (
        <div
          className="flex w-full min-w-0 items-center gap-2 rounded-xl border border-border/60 bg-background/50 px-2.5 py-1.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.18)] transition-colors hover:border-primary/25"
          data-slot="composer-quote-card"
          key={quote.id}
          title={quote.text}
        >
          <Codicon className="shrink-0 text-muted-foreground/75" name="comment-quote" size="0.8125rem" />
          <span className="min-w-0 flex-1 truncate text-[0.8125rem] leading-4 text-foreground/85">
            “{quote.text.replace(/\s+/gu, ' ')}”
          </span>
          {onRemove && (
            <button
              aria-label={t.composer.removeQuote}
              className="grid size-4 shrink-0 place-items-center rounded-full text-muted-foreground/70 transition hover:bg-accent hover:text-foreground"
              data-slot="composer-quote-remove"
              onClick={() => onRemove(quote.id)}
              type="button"
            >
              <Codicon name="close" size="0.625rem" />
            </button>
          )}
        </div>
      ))}
    </div>
  )
}
