/**
 * Mirror of "which chats are mid-turn" to the main process.
 *
 * The renderer is the only side that knows a turn is in flight, and the main
 * process is the only side that can intercept a quit. This module bridges the
 * two: it publishes a small summary on every membership change, and
 * `electron/quit-guard.ts` turns that into the confirmation dialog.
 *
 * Imported for its side effect from `main.tsx`, alongside `store/translucency`.
 */

import { computed } from 'nanostores'

import type { HermesActiveWork, HermesActiveWorkOwner } from '@/global'
import { $sessions } from '@/store/session'
import type { SessionOwnerScope } from '@/store/session-request-router'
import { $workingSessionIds, knownOwnerForSession } from '@/store/session-states'

const $activeWork = computed([$workingSessionIds, $sessions], (workingIds, sessions): HermesActiveWork => {
  const titleById = new Map(sessions.map(session => [session.id, session.title?.trim() ?? '']))

  return {
    count: workingIds.length,
    titles: workingIds.map(id => titleById.get(id) ?? '').filter(Boolean)
  }
})

/** The owner wire the main process resolves to a backend kind (null: unknown, which still holds an update). */
export function activeWorkOwner(owner: SessionOwnerScope): HermesActiveWorkOwner['owner'] {
  if (!owner) {
    return null
  }

  return typeof owner === 'string' ? { profile: owner } : { connectionId: owner.connectionId, profile: owner.profile }
}

const $activeWorkOwners = computed([$workingSessionIds, $sessions], (workingIds, sessions): HermesActiveWorkOwner[] => {
  const titleById = new Map(sessions.map(session => [session.id, session.title?.trim() ?? '']))

  return workingIds.map(id => ({ title: titleById.get(id) ?? '', owner: activeWorkOwner(knownOwnerForSession(id)) }))
})

if (typeof window !== 'undefined') {
  // `$sessions` republishes on unrelated churn (previews, heartbeats), so only
  // send when the summary itself moved — this crosses a process boundary.
  let lastSent = ''
  let lastOwnersSent = ''

  $activeWork.subscribe(work => {
    const next = JSON.stringify(work)

    if (next === lastSent) {
      return
    }

    lastSent = next
    window.hermesDesktop?.setActiveWork?.(work)
  })

  // After the summary (subscription order): main pairs these with that window's report.
  $activeWorkOwners.subscribe(owners => {
    const next = JSON.stringify(owners)

    if (next === lastOwnersSent) {
      return
    }

    lastOwnersSent = next
    window.hermesDesktop?.setActiveWorkOwners?.(owners)
  })
}
