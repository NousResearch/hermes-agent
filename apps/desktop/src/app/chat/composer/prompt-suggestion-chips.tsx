import { memo } from 'react'

import { composerFloatingPill, composerFloatingStrip } from '@/components/chat/composer-dock'
import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

import type { PromptSuggestion } from './prompt-suggestions'

/**
 * Click-to-fill starters above an empty composer.
 *
 * A sibling of the composer, not a child: the pop-out drag region is
 * `absolute inset-0` inside the composer, so anything rendered in there becomes
 * a grab handle (see `composerFloatingStrip`). Centred over the surface, capped
 * so one long title cannot stretch the strip past the input.
 */
export const PromptSuggestionChips = memo(function PromptSuggestionChips({
  items,
  label,
  onPick
}: {
  items: readonly PromptSuggestion[]
  label: string
  onPick: (text: string) => void
}) {
  return (
    <div
      aria-label={label}
      className={cn(composerFloatingStrip, 'justify-center px-[5px] pb-1.5')}
      data-slot="prompt-suggestions"
      role="group"
    >
      {items.map(item => (
        <button
          className={cn(composerFloatingPill, 'max-w-[20rem] min-w-0')}
          key={item.sessionId}
          onClick={() => onPick(item.text)}
          // The label is clipped, so the full ask is the tooltip.
          title={item.text}
          type="button"
        >
          <Codicon className="shrink-0 text-(--ui-text-quaternary)" name="history" size={12} />
          <span className="truncate">{item.text}</span>
        </button>
      ))}
    </div>
  )
})
