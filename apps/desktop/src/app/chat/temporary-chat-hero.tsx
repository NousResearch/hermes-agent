import { IconSpy } from '@tabler/icons-react'

import { useI18n } from '@/i18n'

/**
 * Full-surface empty state for a temporary chat.
 *
 * Shown instead of the usual thread placeholder when a temporary session has
 * no messages yet. The compact bar above the composer is suppressed while this
 * is up (see composer/index.tsx) — two amber blocks saying the same thing at
 * the same moment reads as a bug, not as emphasis.
 *
 * The split is deliberate: at the moment the user opens a temporary chat they
 * have made a privacy decision and deserve an unambiguous confirmation of what
 * it means. Once they start typing, that same message becomes noise competing
 * with their own content, so it shrinks to a one-line bar.
 */
export function TemporaryChatHero() {
  const { t } = useI18n()

  return (
    <div
      className="flex h-full w-full flex-col items-center justify-center gap-4 px-6 text-center"
      data-testid="temporary-chat-hero"
    >
      {/* Muted amber disc rather than a solid fill: this is a reassurance, not
          a warning, and a saturated block at this size reads as an error. */}
      <div className="flex size-16 items-center justify-center rounded-full border border-amber-600/30 bg-amber-500/10 text-amber-700 dark:border-amber-400/25 dark:bg-amber-400/10 dark:text-amber-300">
        <IconSpy aria-hidden className="size-8" stroke={1.5} />
      </div>

      <div className="flex flex-col gap-1.5">
        <h2 className="text-base font-semibold text-amber-900 dark:text-amber-200">
          {t.composer.temporaryHeroTitle}
        </h2>
        {/* Muted foreground, not amber: the heading carries the signal, and a
            second amber line would flatten the hierarchy. */}
        <p className="max-w-sm text-pretty text-[0.8125rem] leading-relaxed text-muted-foreground">
          {t.composer.temporaryHeroBody}
        </p>
      </div>
    </div>
  )
}
