import { useStore } from '@nanostores/react'

import { useComposerScope } from '@/app/chat/composer/scope'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { followUpPreview } from '@/store/composer'

/**
 * The passage waiting to ride the next send: one card above the input, the way
 * a chat app shows the message you are replying to. It rides the composer's
 * scope, so a tile composer shows the tile's passage and nothing else leaks
 * between two composers on screen.
 *
 * Removing it is the only interaction — the passage itself is read-only
 * context. What it becomes is `followUpBlockFromQuote` (store/composer), merged
 * into the outgoing message ahead of what the reader types.
 */
export function FollowUpCard() {
  const scope = useComposerScope()
  const quote = useStore(scope.followUp.$followUp)
  const { t } = useI18n()

  if (!quote) {
    return null
  }

  // Show what the sent bubble will show, so the card is a receipt for the
  // message and not a second, longer version of it.
  const preview = followUpPreview(quote.passage)

  return (
    <div
      className="mx-1 mt-1 flex min-w-0 items-start gap-1.5 rounded-md border-l-2 border-[color-mix(in_srgb,var(--dt-composer-ring)_45%,transparent)] bg-accent/12 py-1 pr-1 pl-2"
      data-slot="composer-follow-up"
    >
      <Codicon className="mt-0.5 shrink-0 text-[0.7rem] text-muted-foreground/80" name="quote" />
      <div className="min-w-0 flex-1">
        <div
          className="text-[0.62rem] text-muted-foreground/80"
          title={quote.source === 'user' ? t.composer.followUp.fromYou : t.composer.followUp.fromAssistant}
        >
          {t.composer.followUp.action}
        </div>
        <div
          className="line-clamp-3 whitespace-pre-wrap text-[0.7rem] text-muted-foreground/88"
          data-slot="composer-follow-up-text"
          title={preview.truncated ? quote.passage : undefined}
        >
          {preview.text}
        </div>
      </div>
      <Button
        aria-label={t.composer.followUp.remove}
        className="h-5 shrink-0 rounded-md px-1.5 text-[0.68rem]"
        onClick={() => scope.followUp.clear()}
        type="button"
        variant="ghost"
      >
        ×
      </Button>
    </div>
  )
}
