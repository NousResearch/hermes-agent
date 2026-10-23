/**
 * Statusbar timer contract (#103123): both surfaces show "focused since".
 * Primary stamps `$sessionStartedAt` on every switch; tiles used to read the
 * row's durable `started_at`, so a day-old tile jumped to 23:03:00. The item is
 * labeled "Focused since" so a long value doesn't read as a stuck turn.
 */

export interface TileSessionFocusStamp {
  since: number
  storedId: string
}

/** Pick the LiveDuration `since` for the statusbar Session item. Never row age. */
export function resolveSessionTimerSince(input: {
  focusedStoredSessionId: null | string
  primaryFocused: boolean
  primarySessionStartedAt: number | null
  tileFocus: null | TileSessionFocusStamp
  /** Runtime-cache anchor for the focused tile, when one exists (#64990). */
  fallbackRuntimeStartedAt?: null | number
}): number | null {
  if (input.primaryFocused) {
    return input.primarySessionStartedAt
  }

  const focused = input.focusedStoredSessionId
  const tile = input.tileFocus

  if (!focused) {
    return null
  }

  // A tile with live runtime state anchors to the session's own runtime
  // start, so a warm-cache reopen or a branch keeps its true elapsed time
  // instead of the parent row's stored age (#64990).
  if (input.fallbackRuntimeStartedAt != null) {
    return input.fallbackRuntimeStartedAt
  }

  if (!tile || tile.storedId !== focused) {
    return null
  }

  return tile.since
}

/**
 * Stamp "focused since" when a non-primary tile is the focused surface.
 * Returns null while primary is focused so the caller leaves the previous
 * stamp alone. Each focus change re-stamps; do not keep a stamp across a
 * detour to primary and back.
 */
export function tileFocusStampOnFocusChange(
  focusedStoredSessionId: null | string,
  selectedStoredSessionId: null | string,
  now: number
): null | TileSessionFocusStamp {
  if (!focusedStoredSessionId || focusedStoredSessionId === selectedStoredSessionId) {
    return null
  }

  return { since: now, storedId: focusedStoredSessionId }
}
