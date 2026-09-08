/**
 * The roster's remembered view filters.
 *
 * A leaf by design: the roster pane reads and writes these, and they know
 * nothing about it. Persisted because a filter you picked is a preference,
 * not a gesture — core's sidebar filters (store/layout.ts) have always
 * survived a reload, and the roster reading as the one place that forgets is
 * the bug this fixes.
 *
 * The search box is deliberately NOT here: a query restored at boot renders
 * an empty-looking roster whose cause is off-screen, which is worse than
 * retyping it.
 */

import { Codecs, persistentAtom } from '@hermes/plugin-sdk'

import type { RosterActivityFilter, RosterKindFilter } from './types'

const KIND_KEY = 'hermes.desktop.botRosterKindFilter'
const ACTIVITY_KEY = 'hermes.desktop.botRosterActivityFilter'
const GATEWAY_KEY = 'hermes.desktop.botRosterGatewayFilter'

/** Decode through the union's own values so a hand-edited or stale key falls
 *  back to 'all' instead of poisoning the filter with an unmatchable value. */
function oneOf<T extends string>(allowed: readonly T[], fallback: T) {
  return {
    decode: (raw: string): T => (allowed.includes(raw as T) ? (raw as T) : fallback),
    encode: (value: T): null | string => value
  }
}

export const $rosterKindFilter = persistentAtom<RosterKindFilter>(
  KIND_KEY,
  'all',
  oneOf(['all', 'bots', 'groups'] as const, 'all')
)

export const $rosterActivityFilter = persistentAtom<RosterActivityFilter>(
  ACTIVITY_KEY,
  'all',
  oneOf(['active', 'all', 'older', 'recent'] as const, 'all')
)

/** Gateway ids are user data, so this one stays free-form text. A gateway that
 *  no longer exists is reconciled by the pane, which falls back to 'all'. */
export const $rosterGatewayFilter = persistentAtom<string>(GATEWAY_KEY, 'all', Codecs.text)

/** Clear all three — the pane's "reset filters" action. */
export function resetRosterFilters() {
  $rosterKindFilter.set('all')
  $rosterActivityFilter.set('all')
  $rosterGatewayFilter.set('all')
}
