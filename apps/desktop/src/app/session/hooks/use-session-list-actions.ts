import { useCallback, useEffect, useRef } from 'react'

import { listAllProfileSessions, listSidebarSessions, type SessionInfo } from '@/hermes'
import { sameCronSignature } from '@/lib/session-signatures'
import {
  isMessagingSource,
  LOCAL_SESSION_SOURCE_IDS,
  MESSAGING_SESSION_SOURCE_IDS,
  normalizeSessionSource
} from '@/lib/session-source'
import {
  $pinnedSessionIds,
  $sessionsLimit,
  $sidebarFiltersActive,
  bumpSessionsLimit,
  raiseSessionsLimit,
  SIDEBAR_FILTERED_PAGE_SIZE,
  SIDEBAR_SESSIONS_PAGE_SIZE
} from '@/store/layout'
import { messagingProfileFor, messagingTotalsKey, normalizeProfileKey } from '@/store/profile'
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
  setSessionProfilesTruncated,
  setSessionProfilesUsage,
  setSessions,
  setSessionsLoading
} from '@/store/session'
import { $workingSessionIds, getRecentlySettledSessionIds } from '@/store/session-states'

import { refreshCronJobs as refreshCronJobsStore } from '../../cron/cron-actions'

// The recents list is local-only: cron rows have their own section, kanban
// dispatcher workers are read on the board, and each messaging platform
// (telegram, discord, …) is fetched separately into its own self-managed
// sidebar section (refreshMessagingSessions). Excluding them here keeps
// "Load more" paging through interactive local chats instead of
// interleaving gateway threads that bury them.
const SIDEBAR_EXCLUDED_SOURCES = ['cron', 'kanban', 'subagent', 'tool', ...MESSAGING_SESSION_SOURCE_IDS]
// The messaging slice is the inverse: drop cron + every local source so only
// external-platform conversations remain, then split per platform in the UI.
const MESSAGING_EXCLUDED_SOURCES = ['cron', ...LOCAL_SESSION_SOURCE_IDS]

// Drop rows the user just deleted/archived: ANY list fetch (full refresh,
// "Load more" paging, a per-platform messaging page, the cron slice) can race
// an in-flight delete RPC, and the backend page still carries the doomed row
// until the DELETE commits — so it flashed back into the sidebar (#50928).
// Honoring the optimistic tombstone at every ingestion point keeps the removal
// stable; the tombstone self-clears once projects.tree confirms the delete,
// and a failed delete untombstones immediately, so nothing is filtered on the
// non-destructive paths.
function dropTombstoned(sessions: SessionInfo[]): SessionInfo[] {
  const tombstones = $removedSessionIds.get()

  return tombstones.size
    ? sessions.filter(s => !tombstones.has(s.id) && !(s._lineage_root_id && tombstones.has(s._lineage_root_id)))
    : sessions
}

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
  const profileScopeRef = useRef(profileScope)
  const loadMoreMessagingRequestRef = useRef<Record<string, number>>({})
  const refreshCronJobsRequestRef = useRef(0)
  const refreshMessagingSessionsRequestRef = useRef(0)
  const refreshSessionsRequestRef = useRef(0)
  profileScopeRef.current = profileScope

  /** Refresh the active profile's messaging-platform sidebar slice. */
  const refreshMessagingSessions = useCallback(async () => {
    const sessionProfile = messagingProfileFor(profileScope)

    // A callback captured before a profile switch may still be queued by an
    // event subscription. Do not let it start a request against the old scope.
    if (messagingProfileFor(profileScopeRef.current) !== sessionProfile) {
      return
    }

    const requestId = refreshMessagingSessionsRequestRef.current + 1
    refreshMessagingSessionsRequestRef.current = requestId

    try {
      const result = await listAllProfileSessions(MESSAGING_SECTION_LIMIT, 1, 'exclude', 'recent', sessionProfile, {
        excludeSources: MESSAGING_EXCLUDED_SOURCES
      })

      if (
        refreshMessagingSessionsRequestRef.current !== requestId ||
        messagingProfileFor(profileScopeRef.current) !== sessionProfile
      ) {
        return
      }

      // Drop any non-messaging source the broad exclude didn't catch (custom
      // sources) — those stay in local recents, not a platform section.
      const rows = dropTombstoned(result.sessions.filter(s => isMessagingSource(s.source)))

      setMessagingSessions(prev => (sameCronSignature(prev, rows) ? prev : rows))
      // Hit the cap → at least one platform may have more on disk than loaded,
      // so platform sections offer their own per-platform "load more".
      setMessagingTruncated(result.sessions.length >= MESSAGING_SECTION_LIMIT)
    } catch {
      // Non-fatal: the messaging sections just stay empty/stale.
    }
  }, [profileScope])

  /** Page one messaging platform without replacing another platform's rows. */
  const loadMoreMessagingForPlatform = useCallback(
    async (platform: string) => {
      const sessionProfile = messagingProfileFor(profileScope)

      if (messagingProfileFor(profileScopeRef.current) !== sessionProfile) {
        return
      }

      const requestKey = messagingTotalsKey(sessionProfile, platform)
      const requestId = (loadMoreMessagingRequestRef.current[requestKey] ?? 0) + 1
      loadMoreMessagingRequestRef.current[requestKey] = requestId

      const inProfile = (s: SessionInfo) =>
        sessionProfile === 'all' || normalizeProfileKey(s.profile) === sessionProfile

      const inPlatform = (s: SessionInfo) => normalizeSessionSource(s.source) === platform && inProfile(s)
      const loaded = $messagingSessions.get().filter(inPlatform).length

      const result = await listAllProfileSessions(
        loaded + SIDEBAR_SESSIONS_PAGE_SIZE,
        1,
        'exclude',
        'recent',
        sessionProfile,
        { source: platform }
      )

      if (
        loadMoreMessagingRequestRef.current[requestKey] !== requestId ||
        messagingProfileFor(profileScopeRef.current) !== sessionProfile
      ) {
        return
      }

      const incoming = dropTombstoned(result.sessions.filter(inPlatform))

      setMessagingSessions(prev => [
        ...prev.filter(s => !inPlatform(s)),
        ...mergeSessionPage(prev.filter(inPlatform), incoming, sessionsToKeep())
      ])

      const total = result.total ?? incoming.length

      setMessagingPlatformTotals(prev => ({ ...prev, [requestKey]: Math.max(total, incoming.length) }))
    },
    [profileScope]
  )

  /** Refresh cron jobs only while the profile that requested them remains active. */
  const refreshCronJobs = useCallback(async () => {
    const sessionProfile = messagingProfileFor(profileScope)

    if (messagingProfileFor(profileScopeRef.current) !== sessionProfile) {
      return
    }

    const requestId = refreshCronJobsRequestRef.current + 1
    refreshCronJobsRequestRef.current = requestId

    try {
      await refreshCronJobsStore(sessionProfile)
    } catch {
      // Non-fatal: the cron section just keeps its last-known jobs.
    }
  }, [profileScope])

  /** Refresh every sidebar session slice without committing an obsolete profile response. */
  const refreshSessions = useCallback(async () => {
    const sessionProfile = messagingProfileFor(profileScope)

    if (messagingProfileFor(profileScopeRef.current) !== sessionProfile) {
      return
    }

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
      // Unified cross-profile list (served read-only off each profile's
      // state.db; no per-profile backend is spawned). Single-profile users get
      // the same rows tagged profile="default".
      // Scope every sidebar slice to the active profile (not always 'all') so a profile
      // with few recent sessions isn't windowed out of the cross-profile
      // recency page and never inherits another profile's cron or messaging
      // sections. ALL_PROFILES remains the explicit unified view.
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

      if (
        refreshSessionsRequestRef.current === requestId &&
        messagingProfileFor(profileScopeRef.current) === sessionProfile
      ) {
        const recents = result.recents

        // Drop rows the user just deleted/archived: a refresh can race an
        // in-flight mutation and the backend page still carries the doomed row.
        // Honoring the optimistic tombstone keeps the removal from flashing back
        // (the tombstone self-clears once projects.tree confirms the delete).
        const incoming = dropTombstoned(recents.sessions)

        // Signature-gate the swap (same pattern as cron/messaging): a refresh
        // that returns content-identical rows must keep the previous array
        // identity, or every sidebar memo keyed on $sessions recomputes and the
        // whole list re-renders once per turn/broadcast for nothing.
        setSessions(prev => {
          const next = mergeSessionPage(prev, incoming, sessionsToKeep())

          return sameCronSignature(prev, next) ? prev : next
        })
        // "Is there another page?" instead of an exact total: the backend
        // reports which profiles filled their window, which costs nothing on
        // top of the rows it already read (the old exact totals ran a COUNT(*)
        // per profile DB on every refresh). Reference-stable when unchanged so
        // the sidebar's group memos don't recompute per refresh.
        setSessionProfilesTruncated(prev => {
          const next = recents.profiles_truncated ?? {}
          const prevKeys = Object.keys(prev)

          return prevKeys.length === Object.keys(next).length && prevKeys.every(key => prev[key] === next[key])
            ? prev
            : next
        })
        // Same identity gate: these totals only move when a session bills, and
        // a fresh object every refresh would repaint every profile header.
        setSessionProfilesUsage(prev => {
          const next = recents.profiles_usage ?? {}
          const prevKeys = Object.keys(prev)

          return prevKeys.length === Object.keys(next).length &&
            prevKeys.every(
              key => prev[key]?.tokens === next[key]?.tokens && prev[key]?.cost_usd === next[key]?.cost_usd
            )
            ? prev
            : next
        })

        // Cron section: latest N cron sessions (kept so a pinned cron run still
        // resolves via sessionByAnyId), signature-gated like above.
        setCronSessions(prev => (sameCronSignature(prev, result.cron.sessions) ? prev : result.cron.sessions))

        // Messaging sections: drop any non-messaging source the broad exclude
        // didn't catch (custom sources stay in local recents), then split per
        // platform in the UI.
        const messagingRows = dropTombstoned(result.messaging.sessions.filter(s => isMessagingSource(s.source)))

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
    if (messagingProfileFor(profileScopeRef.current) === sessionProfile) {
      void refreshCronJobs()
    }
  }, [profileScope, refreshCronJobs])

  const loadMoreSessions = useCallback(async () => {
    bumpSessionsLimit()
    await refreshSessions()
  }, [refreshSessions])

  // A filter searches the loaded page, so switching one on has to deepen the
  // page — otherwise "merged PRs" answers for the last 50 rows and reads as
  // "you only have 6 merged PRs". Clearing the filters hands the window back:
  // the list refreshes on every settled turn, and paying for 300 rows a turn
  // once the view is unfiltered again buys nothing. Whatever the user had
  // paged to by hand is what it returns to.
  const unfilteredLimit = useRef<null | number>(null)

  useEffect(
    () =>
      $sidebarFiltersActive.subscribe(active => {
        if (active) {
          unfilteredLimit.current ??= $sessionsLimit.get()

          if (raiseSessionsLimit(SIDEBAR_FILTERED_PAGE_SIZE)) {
            void refreshSessions()
          }
        } else if (unfilteredLimit.current !== null) {
          const restored = unfilteredLimit.current
          unfilteredLimit.current = null

          if ($sessionsLimit.get() > restored) {
            $sessionsLimit.set(restored)
            void refreshSessions()
          }
        }
      }),
    [refreshSessions]
  )

  return {
    loadMoreMessagingForPlatform,
    loadMoreSessions,
    refreshCronJobs,
    refreshMessagingSessions,
    refreshSessions
  }
}
