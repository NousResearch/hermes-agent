import { type FC, lazy, Suspense, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Popover, PopoverAnchor, PopoverContent } from '@/components/ui/popover'
import { triggerHaptic } from '@/lib/haptics'
import { Plus } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { QUICK_REACTIONS } from '@/store/reactions'
import type { MessageReaction } from '@/types/hermes'

// `frimousse` rides its own lazy chunk (`emoji-picker-full.tsx`): every
// message mounts a ReactionPicker, but the full picker only appears behind
// the quick row's "+", so the library must not sit on the entry graph.
const FullEmojiPicker = lazy(() =>
  import('./emoji-picker-full').then(module => ({ default: module.FullEmojiPicker }))
)

/**
 * The reaction picker — six quick emoji, then "+" for the full set.
 *
 * Rides the shared Popover, so it inherits the app's menu/popover surface
 * treatment rather than inventing a floating pill (DESIGN.md: popovers get one
 * shared shadow + hairline; call sites don't reinvent elevation).
 */
export const ReactionPicker: FC<{
  align?: 'end' | 'start'
  children: React.ReactNode
  onOpenChange: (open: boolean) => void
  onSelect: (emoji: string) => void
  open: boolean
  selected?: string
}> = ({ align = 'end', children, onOpenChange, onSelect, open, selected }) => {
  const [expanded, setExpanded] = useState(false)

  return (
    <Popover
      onOpenChange={next => {
        onOpenChange(next)

        if (!next) {
          // Always reopen on the quick row.
          setExpanded(false)
        }
      }}
      open={open}
    >
      <PopoverAnchor asChild>{children}</PopoverAnchor>
      <PopoverContent
        align={align}
        // Opt this one surface out of the shared popover glass: emoji hover
        // tints at 15% alpha are unreadable over blurred transcript text.
        // Overriding the local surface var keeps the arrow matched for free.
        className={cn('w-auto p-1 [--popover-surface:var(--ui-bg-elevated)]', !expanded && 'flex gap-0.5')}
        onCloseAutoFocus={event => event.preventDefault()}
        side="top"
      >
        {expanded ? (
          <Suspense
            fallback={
              <div className="grid h-72 w-76 place-items-center text-xs text-(--ui-text-tertiary)">
                Loading emoji…
              </div>
            }
          >
            <FullEmojiPicker onSelect={onSelect} />
          </Suspense>
        ) : (
          <>
            {QUICK_REACTIONS.map(emoji => (
              <Button
                aria-label={emoji}
                aria-pressed={selected === emoji}
                className={cn('text-base', selected === emoji && 'bg-(--chrome-action-hover)')}
                key={emoji}
                onClick={() => {
                  triggerHaptic('selection')
                  onSelect(emoji)
                }}
                size="icon-sm"
                variant="ghost"
              >
                {emoji}
              </Button>
            ))}
            <Button aria-label="More emoji" onClick={() => setExpanded(true)} size="icon-sm" variant="ghost">
              <Plus />
            </Button>
          </>
        )}
      </PopoverContent>
    </Popover>
  )
}

/**
 * The reactions a message carries.
 *
 * Flat by design (DESIGN.md: "Flat, not boxed") — no pill, no border, no fill.
 * It reads as quiet metadata in the same register as the message age and the
 * checkpoint row it sits beside. Your own reaction is clickable to retract;
 * the agent's is display-only.
 */
export const ReactionBadge: FC<{
  className?: string
  onRetract?: () => void
  reactions: MessageReaction[]
}> = ({ className, onRetract, reactions }) => {
  if (!reactions.length) {
    return null
  }

  return (
    <span
      className={cn('flex items-center gap-1 text-[0.8125rem] leading-none', className)}
      data-slot="aui_msg-reactions"
    >
      {reactions.map(reaction =>
        reaction.author === 'user' && onRetract ? (
          <button
            aria-label={`Remove ${reaction.emoji} reaction`}
            className="reaction-pop cursor-pointer leading-none transition-transform hover:scale-110 active:scale-95"
            key={`${reaction.author}-${reaction.emoji}`}
            onClick={event => {
              event.preventDefault()
              event.stopPropagation()
              triggerHaptic('selection')
              onRetract()
            }}
            onPointerDown={event => {
              event.preventDefault()
              event.stopPropagation()
            }}
            type="button"
          >
            {reaction.emoji}
          </button>
        ) : (
          <span
            className="reaction-pop leading-none"
            key={`${reaction.author}-${reaction.emoji}`}
            title="Reacted by Hermes"
          >
            {reaction.emoji}
          </span>
        )
      )}
    </span>
  )
}
