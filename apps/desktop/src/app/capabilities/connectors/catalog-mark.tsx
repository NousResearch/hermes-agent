// The one catalog check mark, so the row-card and the dialog cannot disagree
// about what it says or who can hear it.
//
// `Codicon` hardcodes `aria-hidden`, so an `aria-label` on it is inert: the glyph
// is a font ligature with no accessible name and no tooltip a screen reader can
// reach. The label therefore lives on a wrapping span that IS in the
// accessibility tree, and that span is what `Tip` hangs its trigger on.

import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

/** `relative z-10` on purpose: the row-card's name button paints a pseudo element
 *  over the whole card so the row is clickable, and a static glyph underneath it
 *  never receives the pointer that would open its tip. */
export function CatalogMark({ className }: { className?: string }) {
  const { t } = useI18n()
  const label = t.connectorsPage.card.inCatalog

  return (
    <Tip label={label}>
      <span
        aria-label={label}
        className={cn('relative z-10 flex shrink-0 items-center text-(--ui-text-quaternary)', className)}
        role="img"
      >
        <Codicon name="verified-filled" size="0.75rem" />
      </span>
    </Tip>
  )
}
