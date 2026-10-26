import { EmojiPicker } from 'frimousse'
import { type FC, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Popover, PopoverAnchor, PopoverContent } from '@/components/ui/popover'
import { triggerHaptic } from '@/lib/haptics'
import { Plus } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { QUICK_REACTIONS } from '@/store/reactions'
import type { MessageReaction } from '@/types/hermes'

// Served from the app's own origin (vite.config.ts `hermes:emojibase-assets`
// plugin bundles emojibase-data): Electron must work offline, and the app
// should never phone a CDN to draw a picker.
const EMOJIBASE_URL = './emojibase'

// Slack tints its picker cells in a repeating palette (green, blue, yellow,
// pink, brown, purple…) so long scrolls stay scannable. Same trick, in the
// app's own accent idiom (bg-emerald-500/15 etc. are existing patterns).
// Keyed off the emoji's codepoint — deterministic, and stable under
// frimousse's virtualized rows (an index cycle would reshuffle on scroll).
const CELL_TINTS = [
  'hover:bg-emerald-500/15 data-[active]:bg-emerald-500/20',
  'hover:bg-sky-500/15 data-[active]:bg-sky-500/20',
  'hover:bg-amber-500/15 data-[active]:bg-amber-500/20',
  'hover:bg-pink-500/15 data-[active]:bg-pink-500/20',
  'hover:bg-orange-500/15 data-[active]:bg-orange-500/20',
  'hover:bg-violet-500/15 data-[active]:bg-violet-500/20'
] as const

const cellTint = (emoji: string) => CELL_TINTS[(emoji.codePointAt(0) ?? 0) % CELL_TINTS.length]

/** The full emoji picker, revealed behind the quick row's "+". Headless — styled here. */
const FullEmojiPicker: FC<{ onSelect: (emoji: string) => void }> = ({ onSelect }) => (
  <EmojiPicker.Root
    className="flex h-72 w-76 flex-col"
    emojibaseUrl={EMOJIBASE_URL}
    onEmojiSelect={emoji => onSelect(emoji.emoji)}
  >
    {/* Borderless, underline-on-focus — the app's SearchField idiom (DESIGN.md),
        not a boxed search bar. Search matches labels AND emojibase tags
        ("lol" → 😂), which frimousse handles natively. */}
    <EmojiPicker.Search
      autoFocus
      className="mx-1 border-b border-(--ui-stroke-tertiary) bg-transparent px-1 pb-1 text-sm outline-hidden focus:border-(--ui-stroke-secondary)"
      placeholder="Search…"
    />
    <EmojiPicker.Viewport className="relative flex-1 outline-hidden">
      <EmojiPicker.Loading className="absolute inset-0 grid place-items-center text-xs text-(--ui-text-tertiary)">
        Loading emoji…
      </EmojiPicker.Loading>
      <EmojiPicker.Empty className="absolute inset-0 grid place-items-center text-xs text-(--ui-text-tertiary)">
        No emoji found.
      </EmojiPicker.Empty>
      <EmojiPicker.List
        className="select-none pb-1"
        components={{
          CategoryHeader: ({ category, ...props }) => (
            <div
              className="bg-(--ui-bg-elevated) px-1.5 pt-2 pb-1 text-[0.6875rem] text-(--ui-text-tertiary)"
              {...props}
            >
              {category.label}
            </div>
          ),
          Emoji: ({ emoji, ...props }) => (
            <button
              className={cn('grid size-8 shrink-0 place-items-center rounded-md text-lg', cellTint(emoji.emoji))}
              {...props}
            >
              {emoji.emoji}
            </button>
          ),
          Row: ({ children, ...props }) => (
            <div className="flex px-1" {...props}>
              {children}
            </div>
          )
        }}
      />
    </EmojiPicker.Viewport>
  </EmojiPicker.Root>
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
          <FullEmojiPicker onSelect={onSelect} />
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
