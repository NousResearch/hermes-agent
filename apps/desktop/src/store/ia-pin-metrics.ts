import { $apiRequestScope, ambientOwnerConnectionId } from '@/api/client'
import type { SidebarSessionsResponse } from '@/api/sessions'

import { $desktopMetricsGate } from './desktop-metrics'
import { captureIaRecorder } from './ia-metrics'
import { sessionPinId } from './session'
import { $recordsIaMetrics } from './shell-policy'

const bucket = (count: number) => (['zero', 'one', 'two', 'three', 'overflow'] as const)[Math.min(count, 4)]

/** Pins are backfilled beyond page limits. Sample successful loads, never optimistic UI state. */
export function captureIaPinCounts(requestedProfile: string): (page: SidebarSessionsResponse) => void {
  if (!$recordsIaMetrics.get() || $desktopMetricsGate.get() !== 'on') {
    return () => undefined
  }

  const record = captureIaRecorder()
  const profile = $apiRequestScope.get().profile || 'default'
  const owner = ambientOwnerConnectionId()

  return page => {
    const slices = [page.recents, page.cron, page.messaging]

    if (slices.some(slice => slice.failed || slice.retry)) {
      return
    }

    const failed = new Set([
      ...Object.keys(page.storage ?? {}),
      ...Object.keys(page.profiles_failed ?? {}),
      ...(page.errors ?? []).map(error => error.profile),
      ...slices.flatMap(slice => [
        ...Object.keys(slice.profiles_failed ?? {}),
        ...(slice.errors ?? []).map(error => error.profile)
      ])
    ])

    const rows = slices.flatMap(slice => slice.sessions).filter(row => !row.hidden && !row.archived)

    const pins = new Map(
      rows
        .filter(row => row.pinned === true)
        .map(row => [JSON.stringify([row.connection_id ?? owner, row.profile || 'default', sessionPinId(row)]), row])
    )

    // A partial/offline snapshot cannot establish a total; older rows without a pin flag cannot either.
    if (!failed.size && rows.every(row => typeof row.pinned === 'boolean')) {
      record(`ia.pin_count.${bucket(pins.size)}`)
    }

    const owned = (row: (typeof rows)[number]) =>
      (row.connection_id ?? owner) === owner && (row.profile || 'default') === profile

    if (
      (requestedProfile === 'all' || requestedProfile === profile) &&
      !failed.has(profile) &&
      rows.filter(owned).every(row => typeof row.pinned === 'boolean')
    ) {
      record(`ia.profile_pin_count.${bucket([...pins.values()].filter(owned).length)}`)
    }
  }
}
