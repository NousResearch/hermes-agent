import type { Translations } from '@/i18n'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $profileDotStateByScope, type ProfileDotSummary, profileDotSummaryFor } from '@/store/profile-dot-state'

import { sessionDotClassName } from '../session-status-dot'

export function ProfileStatusDot({ summary }: { summary: ProfileDotSummary }) {
  return <span aria-hidden="true" className={sessionDotClassName(summary.state)} data-slot="profile-status-dot" />
}

/** The a11y/tooltip text for one square's summary — every non-zero count,
 *  most urgent first ("1 session needs your answer, 2 unread sessions"). The
 *  counts ride the accessible name so the state is never color-only. */
export function profileStatusLabel(p: Translations['profiles'], summary: ProfileDotSummary): string {
  const parts: string[] = []

  if (summary.needsInputCount > 0) {
    parts.push(p.status.needsInput(summary.needsInputCount))
  }

  if (summary.workingCount > 0) {
    parts.push(p.status.working(summary.workingCount))
  }

  if (summary.unreadCount > 0) {
    parts.push(p.status.unread(summary.unreadCount))
  }

  return parts.join(', ')
}

/** One profile square's rollup of its sessions' shared dot state, keyed by the
 *  (gateway, profile) the square names. `connectionId` is null for this
 *  machine's primary. The interned summaries keep the selector's bail-out
 *  intact: a square re-renders only when its own counts change. */
export function useProfileStatus(
  profile: null | string,
  connectionId: null | string | undefined
): ProfileDotSummary | null {
  return useStoreSelector($profileDotStateByScope, byScope =>
    profile ? (profileDotSummaryFor(byScope, connectionId, profile) ?? null) : null
  )
}

export function combineProfileStatus(summaries: (ProfileDotSummary | undefined)[]): ProfileDotSummary | null {
  const counts = summaries.reduce(
    (total, summary) => ({
      needsInputCount: total.needsInputCount + (summary?.needsInputCount ?? 0),
      workingCount: total.workingCount + (summary?.workingCount ?? 0),
      unreadCount: total.unreadCount + (summary?.unreadCount ?? 0)
    }),
    { needsInputCount: 0, workingCount: 0, unreadCount: 0 }
  )

  if (!counts.needsInputCount && !counts.workingCount && !counts.unreadCount) {
    return null
  }

  return { ...counts, state: counts.needsInputCount ? 'needs-input' : counts.workingCount ? 'working' : 'unread' }
}
