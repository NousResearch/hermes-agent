import { useCallback, useEffect, useRef } from 'react'

import { getCronJobs, listAllProfileSessions, listSidebarSessions, type SessionInfo } from '@/hermes'
import { sameCronSignature } from '@/lib/session-signatures'
import {
  isMessagingSource,
  LOCAL_SESSION_SOURCE_IDS,
  MESSAGING_SESSION_SOURCE_IDS,
  normalizeSessionSource
} from '@/lib/session-source'
import { setCronJobs } from '@/store/cron'
import { $pinnedSessionIds, $sessionsLimit, bumpSessionsLimit, SIDEBAR_SESSIONS_PAGE_SIZE } from '@/store/layout'
import { ALL_PROFILES, normalizeProfileKey } from '@/store/profile'
import { $removedSessionIds } from '@/store/projects'
import {
  $messagingSessions,
  $selectedStoredSessionId,
  $sessions,
  CRON_SECTION_LIMIT,
  mergeSessionPage,
  MESSAGING_SECTION_LIMIT,
  setCronSessions,
  setMessagingPlatformTotals,
  setMessagingSessions,
  setMessagingTruncated,
  setSessionProfileTotals,
  setSessions,
  setSessionsLoading,
  setSessionsTotal
} from '@/store/session'
import { $workingSessionIds, getRecentlySettledSessionIds } from '@/store/session-states'

// The recents list is local-only: cron rows have their own section, and each
// messaging platform (telegram, discord, …) is fetched separately into its own
// self-managed sidebar section (refreshMessagingSessions). Excluding both here
// keeps "Load more" paging through interactive local chats instead of
// interleaving gateway threads that bury them.
const SIDEBAR_EXCLUDED_SOURCES = ['cron', 'subagent', 'tool', ...MESSAGING_SESSION_SOURCE_IDS]
// The messaging slice is the inverse: drop cron + every local source so only
// external-platform conversations remain, then split per platform in the UI.
const MESSAGING_EXCLUDED_SOURCES = ['cron', ...LOCAL_SESSION_SOURCE_IDS]

// Rows a session refresh must preserve even if the aggregator omits them:
// in-flight first turns (message_count 0), pinned rows aged off the page, the
// actively-viewed chat (its "working" flag clears a beat before the aggregator
// sees the persisted row), and sessions whose turn just settled (same race, but
// for a chat the user has already navigated away from). Pass `scope` to only
// keep the active row when it belongs to the profile being paged.
function sessionsToKeep(scope?: string): Set<string> {
  const keep = new Set<string>([
    ...$workingSessionIds.get(),
    ...$pinnedSessionIds.get(),
    ...getRecentlySettledSessionIds()
  ])

  const active = $selectedStoredSessionId.get()

  if (active) {
    const session = scope ? $sessions.get().find(s => s.id === active) : null

    if (!scope || !session || normalizeProfileKey(session.profile) === scope) {
      keep.add(active)
    }
  }

  return keep
}

interface UseSessionListActionsArgs {
  profileScope: string
}

/** Owns the sidebar's session-list fetching + paging: recents, cron runs/jobs,
 *  and the per-platform messaging slices. Returns the callbacks the controller
 *  wires into the sidebar and refresh effects. */
export function useSessionListActions({ profileScope }: UseSessionListActionsArgs) {
  const refreshSessionsRequestRef = useRef(0)
  const profileScopeRef = useRef(profileScope)
  profileScopeRef.current = profileScope

  const sessionProfile = profileScope === ALL_PROFILES ? 'all' : profileScope

  useEffect(() => {
    setMessagingPlatformTotals({})
    setMessagingTruncated(false)
  }, [profileScope])

  // Messaging-platform sessions as their own slice, fetched separately from
  // local recents so each platform renders a self-managed section and never
  // competes with local chats for the recents page budget. One combined fetch
  // seeds every platform; the sidebar splits the rows per source.
  const refreshMessagingSessions = useCallback(async () => {
    const requestProfile = profileScope === ALL_PROFILES ? 'all' : profileScope

    try {
      const result = await listAllProfileSessions(MESSAGING_SECTION_LIMIT, 1, 'exclude', 'recent', requestProfile, {
        excludeSources: MESSAGING_EXCLUDED_SOURCES
      })

      if (profileScopeRef.current !== profileScope) {
        return
      }

      // Drop any non-messaging source the broad exclude didn't catch (custom
      // sources) — those stay in local recents, not a platform section.
      const rows = result.sessions.filter(
        s =>
          isMessagingSource(s.source) &&
          (requestProfile === 'all' || normalizeProfileKey(s.profile) === requestProfile)
      )

      setMessagingSessions(prev => (sameCronSignature(prev, rows) ? prev : rows))
      // Hit the cap → at least one platform may have more on disk than loaded,
      // so platform sections offer their own per-platform "load more".
      setMessagingTruncated(result.sessions.length >= MESSAGING_SECTION_LIMIT)
    } catch {
      // Non-fatal: the messaging sections just stay empty/stale.
    }
  }, [profileScope])

  // Page a single platform's section independently (mirrors the per-profile
  // pager): fetch that source's next window and merge it back in place, leaving
  // every other platform's rows untouched. Resolves the platform's exact total.
  const loadMoreMessagingForPlatform = useCallback(async (platform: string) => {
    const requestProfile = profileScope === ALL_PROFILES ? 'all' : profileScope
    const inPlatform = (s: SessionInfo) => normalizeSessionSource(s.source) === platform

    const inScope = (s: SessionInfo) =>
      requestProfile === 'all' || normalizeProfileKey(s.profile) === requestProfile

    const loaded = $messagingSessions.get().filter(s => inPlatform(s) && inScope(s)).length

    const result = await listAllProfileSessions(
      loaded + SIDEBAR_SESSIONS_PAGE_SIZE,
      1,
      'exclude',
      'recent',
      requestProfile,
      { source: platform }
    )

    if (profileScopeRef.current !== profileScope) {
      return
    }

    const incoming = result.sessions.filter(s => inPlatform(s) && inScope(s))

    setMessagingSessions(prev => [
      ...prev.filter(s => !inPlatform(s)),
      ...mergeSessionPage(prev.filter(inPlatform), incoming, sessionsToKeep())
    ])

    const total = result.total ?? incoming.length
    setMessagingPlatformTotals(prev => ({ ...prev, [platform]: Math.max(total, incoming.length) }))
  }, [profileScope])

  // Cron *jobs* drive the sidebar "Cron jobs" section. Jobs are created
  // synchronously (agent tool call or the cron UI), so refreshing here right
  // after an agent turn surfaces a new job immediately; the interval poll keeps
  // next-run/state fresh as the scheduler advances them. Jobs live per-profile
  // on disk and the list endpoint aggregates 'all' by default, so scope the
  // fetch to the sidebar's profile scope — a concrete profile sees only its
  // own jobs; ALL_PROFILES keeps the unified view.
  const refreshCronJobs = useCallback(async () => {
    try {
      const jobs = await getCronJobs(profileScope === ALL_PROFILES ? 'all' : profileScope)

      setCronJobs(jobs)
    } catch {
      // Non-fatal: the cron section just keeps its last-known jobs.
    }
  }, [profileScope])

  const refreshSessions = useCallback(async () => {
    const requestId = refreshSessionsRequestRef.current + 1
    refreshSessionsRequestRef.current = requestId
    // The loading flag exists to drive the initial skeletons (they only render
    // while the list is empty). Turn-complete / reconnect refreshes over a
    // populated list used to flip it true→false anyway, churning every
    // $sessionsLoading subscriber twice per turn for no visible change.
    const showLoading = $sessions.get().length === 0

    if (showLoading) {
      setSessionsLoading(true)
    }

    try {
      const limit = $sessionsLimit.get()

      // Require at least one message so abandoned/empty "Untitled" drafts (one
      // was created per TUI/desktop launch before the lazy-create fix) don't
      // clutter the sidebar.
      // All Profiles is the only unified cross-profile workspace. Concrete
      // profile workspaces scope every sidebar slice to that profile.
      // Batched: one request opens each profile DB once and returns all three
      // source-scoped slices, instead of three separate listAllProfileSessions
      // calls that each reopened + re-counted every profile DB per refresh.
      const result = await listSidebarSessions({
        recentsProfile: sessionProfile,
        recentsLimit: limit,
        recentsExclude: SIDEBAR_EXCLUDED_SOURCES,
        cronLimit: CRON_SECTION_LIMIT,
        messagingLimit: MESSAGING_SECTION_LIMIT,
        messagingExclude: MESSAGING_EXCLUDED_SOURCES
      })

      if (refreshSessionsRequestRef.current === requestId && profileScopeRef.current === profileScope) {
        const recents = result.recents

        const inScope = (s: SessionInfo) =>
          sessionProfile === 'all' || normalizeProfileKey(s.profile) === sessionProfile

        // Drop rows the user just deleted/archived: a refresh can race an
        // in-flight mutation and the backend page still carries the doomed row.
        // Honoring the optimistic tombstone keeps the removal from flashing back
        // (the tombstone self-clears once projects.tree confirms the delete).
        const tombstones = $removedSessionIds.get()

        const scopedRecents = recents.sessions.filter(inScope)

        const incoming = tombstones.size
          ? scopedRecents.filter(
              s => !tombstones.has(s.id) && !(s._lineage_root_id && tombstones.has(s._lineage_root_id))
            )
          : scopedRecents

        // Signature-gate the swap (same pattern as cron/messaging): a refresh
        // that returns content-identical rows must keep the previous array
        // identity, or every sidebar memo keyed on $sessions recomputes and the
        // whole list re-renders once per turn/broadcast for nothing.
        setSessions(prev => {
          const next = mergeSessionPage(prev, incoming, sessionsToKeep())

          return sameCronSignature(prev, next) ? prev : next
        })
        setSessionsTotal(typeof recents.total === 'number' ? recents.total : recents.sessions.length)
        setSessionProfileTotals(prev => {
          const next = recents.profile_totals ?? {}
          const prevKeys = Object.keys(prev)

          return prevKeys.length === Object.keys(next).length && prevKeys.every(key => prev[key] === next[key])
            ? prev
            : next
        })

        // Cron section: latest N cron sessions (kept so a pinned cron run still
        // resolves via sessionByAnyId), signature-gated like above.
        const cronRows = result.cron.sessions.filter(inScope)
        setCronSessions(prev => (sameCronSignature(prev, cronRows) ? prev : cronRows))

        // Messaging sections: drop any non-messaging source the broad exclude
        // didn't catch (custom sources stay in local recents), then split per
        // platform in the UI.
        const messagingRows = result.messaging.sessions.filter(s => inScope(s) && isMessagingSource(s.source))

        setMessagingSessions(prev => (sameCronSignature(prev, messagingRows) ? prev : messagingRows))

        // Hit the cap → at least one platform may have more on disk than loaded.
        setMessagingTruncated(result.messaging.sessions.length >= MESSAGING_SECTION_LIMIT)
      }
    } finally {
      if (showLoading && refreshSessionsRequestRef.current === requestId) {
        setSessionsLoading(false)
      }
    }

    // Cron *jobs* are a distinct API (getCronJobs), not a session slice.
    void refreshCronJobs()
  }, [profileScope, refreshCronJobs, sessionProfile])

  const loadMoreSessions = useCallback(async () => {
    bumpSessionsLimit()
    await refreshSessions()
  }, [refreshSessions])

  // ALL-profiles view pages one profile at a time: fetch that profile's next
  // page and merge it in place, leaving every other profile's rows untouched.
  const loadMoreSessionsForProfile = useCallback(async (profile: string) => {
    const key = normalizeProfileKey(profile)
    const inKey = (s: SessionInfo) => normalizeProfileKey(s.profile) === key
    const loaded = $sessions.get().filter(inKey).length

    const result = await listAllProfileSessions(loaded + SIDEBAR_SESSIONS_PAGE_SIZE, 1, 'exclude', 'recent', key, {
      excludeSources: SIDEBAR_EXCLUDED_SOURCES
    })

    const keep = sessionsToKeep(key)

    setSessions(prev => [
      ...prev.filter(s => !inKey(s)),
      ...mergeSessionPage(prev.filter(inKey), result.sessions, keep)
    ])

    const total = result.profile_totals?.[key] ?? result.total ?? result.sessions.length
    setSessionProfileTotals(prev => ({ ...prev, [key]: Math.max(total, result.sessions.length) }))
  }, [])

  return {
    loadMoreMessagingForPlatform,
    loadMoreSessions,
    loadMoreSessionsForProfile,
    refreshCronJobs,
    refreshMessagingSessions,
    refreshSessions
  }
}
