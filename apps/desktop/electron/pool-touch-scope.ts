export function poolTouchKeys(scope: unknown): string[] {
  const key = String(scope ?? '').trim()

  if (!key) {
    return []
  }

  const localPrefix = 'conn:local::'

  if (key.startsWith(localPrefix)) {
    const delegatedProfile = key.slice(localPrefix.length)

    if (delegatedProfile) {
      return [key, delegatedProfile]
    }
  }

  return [key]
}

/**
 * Event-driven release for #102187: the window's active route moved away from
 * this scope (profile switch), so its backend is no longer plausibly live even
 * though its last keepalive touch is still inside the fresh window. Rewind
 * (don't zero) its freshness: the next spawn's LRU eviction reclaims the slot
 * immediately, while the idle reaper still grants its normal tail grace and
 * any genuine later use re-touches. Already-stale entries are untouched.
 * Multi-window safe: another window's keepalive simply re-freshens a backend
 * that is still in use.
 */
export function markPoolScopeReleased(
  pool: Map<string, { lastActiveAt?: null | number }>,
  scope: unknown,
  now: number,
  freshMs: number
): void {
  const staleAt = now - freshMs - 1

  for (const key of poolTouchKeys(scope)) {
    const entry = pool.get(key)

    if (entry && (entry.lastActiveAt || 0) > staleAt) {
      entry.lastActiveAt = staleAt
    }
  }
}
