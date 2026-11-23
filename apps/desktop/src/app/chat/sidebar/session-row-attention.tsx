import type { ReactNode } from 'react'

import { useI18n } from '@/i18n'
import { type SessionDotState, showsRunningArc } from '@/store/session-dot-state'

// Attention cues on a sidebar row, beside (not instead of) the status dot. A
// 6px dot alone did not carry "this one is waiting on you": idle rows paint
// the same dot in their project colour, so amber read as one more tint. These
// cues are only ever drawn for the states a person has to act on or has not
// seen yet, so a quiet list stays quiet.

/** Trailing "Needs you" pill for a row whose turn is blocked on the owner
 *  (a clarify/approval prompt, or a paused queue). Nothing otherwise. */
export function NeedsYouBadge({ state }: { state: SessionDotState }) {
  const { t } = useI18n()

  if (state !== 'needs-input') {
    return null
  }

  return (
    <span
      className="pointer-events-none shrink-0 whitespace-nowrap rounded-full bg-amber-500/15 px-1.5 text-[0.625rem] font-medium leading-4 text-amber-600 dark:text-amber-400"
      data-needs-you=""
    >
      {t.sidebar.row.needsYouBadge}
    </span>
  )
}

/** Amber bar on the row's leading edge, same condition as the pill: it is
 *  the cue that still reads when the sidebar is narrow and the pill is far
 *  from where the eye scans (the titles). */
export function NeedsYouEdge({ state }: { state: SessionDotState }) {
  if (state !== 'needs-input') {
    return null
  }

  return (
    <span
      aria-hidden="true"
      className="pointer-events-none absolute inset-y-1 left-0 w-0.5 rounded-full bg-amber-500"
      data-needs-you-edge=""
    />
  )
}

/** Title weight: unread rows are semibold and brighter, read rows normal.
 *  Either source counts: the row's backend read state (`unread` prop) or the
 *  resolved dot state (a turn that finished in the background). */
export function rowTitleWeight(unread: boolean, state: SessionDotState): string {
  return unread || state === 'unread' ? 'font-semibold text-foreground' : 'font-normal'
}

/** The row's leading state layers: the running arc (unchanged) and the amber
 *  needs-you edge, rendered as one element of the row. */
export function RowStateCues({ state }: { state: SessionDotState }) {
  return (
    <>
      {showsRunningArc(state) && <span aria-hidden="true" className="arc-border arc-row" />}
      <NeedsYouEdge state={state} />
    </>
  )
}

/** Seed for the row's trailing chip slot: the "Needs you" pill comes first,
 *  so it reads before the profile/PR chips and the figures. */
export function needsYouTrailing(state: SessionDotState): { key: string; node: ReactNode }[] {
  return state === 'needs-input' ? [{ key: 'needs-you', node: <NeedsYouBadge state={state} /> }] : []
}
