import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useEffect, useState } from 'react'

import { getUsageAnalytics, type ProfileScope, profileScopeKey } from '@/hermes'

// Per-tool call counts come from a 365-day message scan — heavy, and purely
// cosmetic (Toolsets usage badges). Cache the result module-wide with a TTL so
// bouncing between tabs/pages doesn't re-run the scan every time. Keyed by
// the Capabilities scope profile: analytics are profile-scoped, so a scope or
// app-profile switch must not show the previous profile's counts.
const TOOL_CALLS_TTL_MS = 10 * 60 * 1000
const toolCallsCache = new Map<string, { at: number; value: Record<string, number> }>()

// Bumped once a forced pull has landed in the cache, so badges already on
// screen pick the fresh counts up instead of sitting out the TTL.
const $toolCallsRefreshed = atom(0)

async function loadToolCalls(
  scopeKey: string,
  scopeProfile: ProfileScope,
  force = false
): Promise<Record<string, number>> {
  const cached = toolCallsCache.get(scopeKey)

  if (!force && cached && Date.now() - cached.at < TOOL_CALLS_TTL_MS) {
    return cached.value
  }

  const analytics = await getUsageAnalytics(365, scopeProfile)

  const value = Object.fromEntries((analytics.tools ?? []).map(e => [e.tool, e.count]))

  toolCallsCache.set(scopeKey, { at: Date.now(), value })

  return value
}

/** The page refresh's analytics leg (`useRefreshHotkey`): the one time the TTL
 *  is bypassed — but only if the badges have been on screen at least once;
 *  otherwise let the lazy load pick the counts up when Toolsets is first
 *  shown. */
export async function refreshToolCalls(profile: ProfileScope): Promise<void> {
  if (toolCallsCache.size === 0) {
    return
  }

  await loadToolCalls(profileScopeKey(profile), profile, true).catch(() => undefined)
  $toolCallsRefreshed.set($toolCallsRefreshed.get() + 1)
}

/**
 * tool name -> call count over the analytics window. `null` = still loading
 * (badges show skeletons); `{}` = loaded empty / unavailable backend.
 *
 * Mounting the Toolsets tab is what asks for the scan, so Skills and MCP never
 * pay for it and it can't starve the MCP tab's config load. A scope change
 * remounts the tab, so a slow load from the previous scope can never land on
 * the new one's badges.
 */
export function useToolCalls(profile: ProfileScope): null | Record<string, number> {
  const refreshed = useStore($toolCallsRefreshed)
  const [toolCalls, setToolCalls] = useState<null | Record<string, number>>(null)
  const scopeKey = profileScopeKey(profile)

  useEffect(() => {
    let cancelled = false

    loadToolCalls(scopeKey, profile)
      .then(value => !cancelled && setToolCalls(value))
      .catch(() => !cancelled && setToolCalls({}))

    return () => void (cancelled = true)
  }, [profile, refreshed, scopeKey])

  return toolCalls
}
