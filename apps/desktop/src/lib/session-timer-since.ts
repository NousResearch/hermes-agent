/**
 * Statusbar "Session" timer contract (#103123).
 *
 * Primary focus stamps `$sessionStartedAt` with Date.now() on every switch
 * ("focused since"). Tile focus used to read the durable DB `started_at`, so
 * the same control jumped from `0:05` to `23:03:00` when clicking a day-old
 * tile. Both surfaces now share the focus-since meaning.
 */

export interface TileSessionFocusStamp {
  since: number
  storedId: string
}

/** Pick the LiveDuration `since` for the statusbar Session item. */
export function resolveSessionTimerSince(input: {
  focusedStoredSessionId: null | string
  primaryFocused: boolean
  primarySessionStartedAt: number | null
  tileFocus: null | TileSessionFocusStamp
}): number | null {
  if (input.primaryFocused) {
    return input.primarySessionStartedAt
  }

  const focused = input.focusedStoredSessionId
  const tile = input.tileFocus

  if (!focused || !tile || tile.storedId !== focused) {
    return null
  }

  return tile.since
}

/**
 * Stamp (or keep) the tile focus-since clock when the focused surface is a
 * non-primary tile. Re-focusing a different tile resets the clock the same way
 * primary activation re-stamps `$sessionStartedAt`.
 */
export function nextTileSessionFocusStamp(
  previous: null | TileSessionFocusStamp,
  focusedStoredSessionId: null | string,
  primaryFocused: boolean,
  now: number
): null | TileSessionFocusStamp {
  if (primaryFocused || !focusedStoredSessionId) {
    return previous
  }

  if (previous?.storedId === focusedStoredSessionId) {
    return previous
  }

  return { since: now, storedId: focusedStoredSessionId }
}
