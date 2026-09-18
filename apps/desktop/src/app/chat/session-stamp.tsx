/**
 * The stamp chips — small tinted labels painted next to a session's title.
 *
 * Every surface that shows a session title reads the same chips off the row's
 * labels (sidebar rows, session tabs, the tab rail, the ⌘ switcher), so a stamp
 * cannot mean one thing in the list and another in a tab. Hue carries the meaning
 * at a glance; a label the app has never seen still renders, in the theme accent.
 *
 * A session can carry up to three labels (store/session-stamp owns that list);
 * the group renders them in order; the row's title truncates first, so a chip
 * is never the thing that gets clipped.
 */

import { useStore } from '@nanostores/react'

import { useStoreSelector } from '@/lib/use-session-slice'
import { cn } from '@/lib/utils'
import { $selectedStoredSessionId, $sessions, sessionMatchesStoredId } from '@/store/session'
import {
  $sessionStamps,
  $stampColorOverrides,
  isEmojiStamp,
  normalizeSessionStamp,
  SESSION_STAMP_LIMIT,
  stampColorFor,
  stampLabels
} from '@/store/session-stamp'
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
//
// `normal-case` is load-bearing: a tab label wraps its text in a span that
// UPPERCASES it (PaneTabLabel), so without this the same stamp reads "MRG" in
// the strip and "mrg" in the session list. Text transforms are inherited, so the
// chip opts out at its own root.
const STAMP_CHIP =
  'inline-flex max-w-28 shrink-0 items-center truncate rounded-[3px] bg-[color-mix(in_srgb,currentColor_14%,transparent)] px-1 text-[0.5625rem] font-medium leading-[1.6] tracking-[0.01em] normal-case'

// An EMOJI stamp is the glyph alone: bigger, no wash, no hue. A colour emoji is
// drawn by the platform's own emoji font (Apple Color Emoji on macOS), which
// ignores `color` — so a tinted capsule behind it would claim a colour choice
// the user cannot see, and 9px text sizing would shrink a stamp that IS the
// whole mark. `max-w-28` + truncate still rail an absurd multi-emoji label.
const STAMP_EMOJI_CHIP =
  'inline-flex max-w-28 shrink-0 items-center truncate px-0.5 text-[0.8125rem] leading-[1.15] normal-case'

// The group owning the room the chips may take. No width cap of its own: wherever
// stamps share a row with a title (sidebar rows, session tabs, the ⌘ switcher) the
// TITLE yields first, so a chip is never clipped by the surface around it — only the
// chip's own max-width rails against an absurd custom label.
const STAMP_GROUP = 'inline-flex min-w-0 shrink-0 items-center gap-0.5'

export function SessionStamp({ className, stamp }: { className?: string; stamp: null | string | undefined }) {
  // Hooks precede the early return: an unstamped row still calls the store.
  const overrides = useStore($stampColorOverrides)
  const label = normalizeSessionStamp(stamp)

  if (!label) {
    return null
  }

  const emoji = isEmojiStamp(label)
  // An emoji wears no colour: the glyph is the stamp, and the platform's colour
  // font ignores `color` anyway (the menu hides the choice for the same reason).
  const color = emoji ? null : stampColorFor(label, overrides)

  return (
    <span
      className={cn(emoji ? STAMP_EMOJI_CHIP : STAMP_CHIP, !color && !emoji && stampHueClass(label), className)}
      data-session-stamp={label}
      // A picked colour rides `currentColor`, so the chip's wash
      // (`color-mix(currentColor …)`) follows the label instead of the theme.
      style={color ? { color } : undefined}
      title={label}
    >
      {label}
    </span>
  )
}

/**
 * Every stamp a session carries, in order, as ONE group.
 *
 * Rendered as a group (rather than one component per call site) so the list, the
 * tabs and the switcher cannot disagree about how many chips fit or how they
 * clip. Adding a label past the cap is impossible by construction.
 */
export function SessionStamps({ className, stamps }: { className?: string; stamps: readonly string[] }) {
  if (!stamps.length) {
    return null
  }

  return (
    <span className={cn(STAMP_GROUP, className)}>
      {stamps.slice(0, SESSION_STAMP_LIMIT).map(label => (
        <SessionStamp key={label} stamp={label} />
      ))}
    </span>
  )
}

/** The hue a title wears when the user has not picked one for it; a label the
 *  app has never seen takes the accent. Shared with the menu's colour
 *  affordance, so the dot there shows the colour the chip will actually paint. */
export function stampHueClass(label: string): string {
  return STAMP_HUES[label.toLowerCase()] ?? 'text-(--ui-accent)'
}

// The separator the tab's label selector rides: `useStoreSelector` must return a
// primitive (or a referentially stable value) or every tab repaints on every
// list poll, and a label list is a fresh array on each poll. A NUL byte cannot
// appear in a label (the normalizer strips control characters server-side).
const LABEL_KEY_SEP = '\u0000'

/** The MAIN tab's pane id (app/contrib/controller). It is not one tile among
 *  others but the window's PRIMARY session, which is why it carries no id of
 *  its own — see `SessionTabStamp`. */
const MAIN_PANE_ID = 'workspace'

/**
 * The stamps for a PANE, for the tab strip.
 *
 * The strip passes the pane id and this resolves the session behind it — a
 * `session-tile:<storedId>` pane IS that session (store/session-states), and the
 * `workspace` pane is the window's primary session — so no title plumbing is
 * needed and any other tab renders nothing. The id may be an older segment of a
 * compression chain, which is why the lookup goes through the lineage-aware
 * stamp map rather than the live id.
 */
export function SessionTabStamp({ className, paneId }: { className?: string; paneId: string }) {
  // The primary session belongs to the WINDOW, not to a pane: the main tab is
  // titled and dotted off `$selectedStoredSessionId`
  // (app/contrib/controller -> syncWorkspaceTitle), so its chips read the same
  // source and cannot disagree with the tab they sit on. Null on a fresh draft,
  // which is a tab with no session behind it and therefore no chips.
  const primaryStoredId = useStore($selectedStoredSessionId)

  const storedSessionId = paneId.startsWith(TILE_PANE_PREFIX)
    ? paneId.slice(TILE_PANE_PREFIX.length)
    : paneId === MAIN_PANE_ID
      ? primaryStoredId
      : null

  // The LIST's own row first: that is the exact list the sidebar paints, so a tab
  // can never disagree with the session list (including a label's CASE). The
  // derived map is the fallback for a session that is only in a cron or messaging
  // list — but a lineage id can be claimed by more than one row there, and
  // whichever was mapped last used to win.
  const ownLabels = useStoreSelector($sessions, rows => {
    const row = storedSessionId
      ? rows.find(candidate => sessionMatchesStoredId(candidate, storedSessionId))
      : undefined

    return row ? stampLabels(row).join(LABEL_KEY_SEP) : ''
  })

  // Same reason: the map is rebuilt on every poll, so the selector hands back the
  // joined key and the array is rebuilt only where it is actually painted.
  const mappedKey = useStoreSelector($sessionStamps, stamps => {
    const labels = storedSessionId ? stamps.get(storedSessionId) : undefined

    return labels ? labels.join(LABEL_KEY_SEP) : ''
  })

  const labels = ownLabels
    ? ownLabels.split(LABEL_KEY_SEP)
    : mappedKey
      ? mappedKey.split(LABEL_KEY_SEP)
      : []

  return <SessionStamps className={className} stamps={labels} />
}
