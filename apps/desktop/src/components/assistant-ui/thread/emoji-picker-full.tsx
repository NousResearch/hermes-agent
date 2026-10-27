/**
 * FULL EMOJI PICKER — split out of `message-reactions.tsx` so `frimousse`
 * stays off the entry graph.
 *
 * Every message mounts a `ReactionPicker`, but the full picker only appears
 * behind the quick row's "+". This module (and frimousse with it) loads on
 * first expand via `React.lazy` + `Suspense` in the picker — the quick row
 * paints immediately either way.
 */

import { EmojiPicker } from 'frimousse'
import type { FC } from 'react'

import { cn } from '@/lib/utils'

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
export const FullEmojiPicker: FC<{ onSelect: (emoji: string) => void }> = ({ onSelect }) => (
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
