/** Compact Plugins header lives in a HALF_COL `w-36` cell. `max-w-64` (256px)
 *  is wider than that cell and lets the trigger grow the window; truncate
 *  inside `min-w-0` instead. */
export const COMPACT_SCOPE_TRIGGER_CLASS = 'h-6 min-w-0 w-full px-2 truncate'

/** Cap the open list to the remaining viewport — do not inherit the
 *  overflowing trigger width. Passed only from the compact caller. */
export const COMPACT_SCOPE_CONTENT_CLASS = 'max-w-(--radix-select-available-width)'

export function rosterScopeLabel(
  agent: { connectionId: string; connectionLabel: string; profile: string },
  activeId: string,
  compact: boolean
): string {
  const profile = agent.profile.trim() || 'default'

  if (compact) {
    return profile === 'default' ? agent.connectionLabel : `${profile} — ${agent.connectionLabel}`
  }

  const base = `${profile} — ${agent.connectionLabel}`
  return agent.connectionId === activeId ? `${base} (current)` : base
}
