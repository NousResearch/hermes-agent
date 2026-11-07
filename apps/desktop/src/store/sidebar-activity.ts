import { computed } from 'nanostores'

import { stableRecord } from '@/lib/stable-array'

import { $backgroundStatusBySession, type ComposerStatusItem, isAwaitedBackgroundWork } from './composer-status'
import { $sessions, lineageAliases } from './session'
import { $sessionDotStateById } from './session-dot-state'
import { $sessionStates } from './session-states'

export type SidebarActivity = ComposerStatusItem

let previous: Readonly<Record<string, SidebarActivity>> = {}

/** Project awaited background work only, using the shared status priority.
 * No transcript scan or prose classification: active agents keep their arc. */
export const $sidebarActivityById = computed(
  [$sessionStates, $backgroundStatusBySession, $sessions, $sessionDotStateById],
  (states, items, sessions, statuses) => {
    const next: Record<string, SidebarActivity> = {}

    for (const [runtimeId, processes] of Object.entries(items)) {
      const state = states[runtimeId]

      const activity = processes.find(isAwaitedBackgroundWork)

      if (!activity) {
        continue
      }

      for (const alias of lineageAliases(state?.storedSessionId ?? runtimeId, sessions)) {
        if (statuses[alias] === 'background' && !next[alias]) {
          next[alias] = activity
        }
      }
    }

    return (previous = stableRecord(previous, next))
  }
)
