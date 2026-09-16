/**
 * The stamp chip — one small tinted label painted next to a session's title.
 *
 * Every surface that shows a session title reads this same chip off the row's
 * `stamp` (sidebar rows, session tabs, the tab rail), so a stamp cannot mean one
 * thing in the list and another in a tab. Hue carries the meaning at a glance; a
 * label the app has never seen still renders, in the theme accent.
 */

import { useStoreSelector } from '@/lib/use-session-slice'
import { cn } from '@/lib/utils'
import { $sessionStamps, normalizeSessionStamp } from '@/store/session-stamp'
import { TILE_PANE_PREFIX } from '@/store/session-states'

// Preset -> hue. The presets themselves are data in store/session-stamp; this is
// only how the five of them look. Anything else wears the accent so a custom
// label reads as the user's own, never as a preset it isn't.
const STAMP_HUES: Record<string, string> = {
  handoff: 'text-(--ui-purple)',
  hold: 'text-(--ui-red)',
  merged: 'text-(--ui-green)',
  review: 'text-(--ui-blue)',
  wip: 'text-(--ui-yellow)'
}

// A wash of the chip's OWN colour rather than a fixed surface token: the chip
// has to read as a stamp under every theme and every row state (hover, active,
// selected) without adding a token per stamp. `max-w` + truncate keep a long
// custom stamp from eating the title's line.
const STAMP_CHIP =
  'inline-flex max-w-28 shrink-0 items-center truncate rounded-[3px] bg-[color-mix(in_srgb,currentColor_14%,transparent)] px-1 text-[0.5625rem] font-medium leading-[1.6] tracking-[0.01em]'

export function SessionStamp({ className, stamp }: { className?: string; stamp: null | string | undefined }) {
  const label = normalizeSessionStamp(stamp)

  if (!label) {
    return null
  }

  return (
    <span
      className={cn(STAMP_CHIP, STAMP_HUES[label.toLowerCase()] ?? 'text-(--ui-accent)', className)}
      data-session-stamp={label}
      title={label}
    >
      {label}
    </span>
  )
}

/**
 * The stamp for a PANE, for the tab strip.
 *
 * The strip passes the pane id and this resolves the session behind it — a
 * `session-tile:<storedId>` pane IS that session (store/session-states) — so no
 * title plumbing is needed and any other tab renders nothing. The id may be an
 * older segment of a compression chain, which is why the lookup goes through the
 * lineage-aware stamp map rather than the live id.
 */
export function SessionTabStamp({ className, paneId }: { className?: string; paneId: string }) {
  const storedSessionId = paneId.startsWith(TILE_PANE_PREFIX) ? paneId.slice(TILE_PANE_PREFIX.length) : null
  const stamp = useStoreSelector($sessionStamps, stamps => (storedSessionId ? stamps.get(storedSessionId) : undefined))

  return <SessionStamp className={className} stamp={stamp} />
}
