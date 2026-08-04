import { useCallback, useRef } from 'react'

import {
  BULK_DELETE_MAX_IDS,
  bulkDeleteSessions,
  getCronJobs,
  listAllProfileSessions,
  listSidebarSessions,
  type SessionInfo
} from '@/hermes'
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
  sessionPinId,
  setCronSessions,
  setMessagingPlatformTotals,
  setMessagingSessions,
  setMessagingTruncated,
  setSessionProfilesTruncated,
  setSessions,
  setSessionsLoading
} from '@/store/session'
import { $workingSessionIds, getRecentlySettledSessionIds } from '@/store/session-states'

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

// Upper bound on the page-and-delete loop in clearAllSessions. Each pass clears
// up to BULK_DELETE_MAX_IDS rows, so this caps a single "Delete all" at ~500k
// chats - far past any real history - while guaranteeing the loop terminates if
// the backend ever stops actually deleting.
const SESSION_CLEAR_MAX_PAGES = 1_000

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

  // Messaging-platform sessions as their own slice, fetched separately from
  // local recents so each platform renders a self-managed section and never
  // competes with local chats for the recents page budget. One combined fetch
  // seeds every platform; the sidebar splits the rows per source.
  const refreshMessagingSessions = useCallback(async () => {
    try {
      const result = await listAllProfileSessions(MESSAGING_SECTION_LIMIT, 1, 'exclude', 'recent', 'all', {
        excludeSources: MESSAGING_EXCLUDED_SOURCES
      })

      // Drop any non-messaging source the broad exclude didn't catch (custom
      // sources) — those stay in local recents, not a platform section.
      const rows = result.sessions.filter(s => isMessagingSource(s.source))

      setMessagingSessions(prev => (sameCronSignature(prev, rows) ? prev : rows))
      // Hit the cap → at least one platform may have more on disk than loaded,
      // so platform sections offer their own per-platform "load more".
      setMessagingTruncated(result.sessions.length >= MESSAGING_SECTION_LIMIT)
    } catch {
      // Non-fatal: the messaging sections just stay empty/stale.
    }
  }, [])

  // Page a single platform's section independently (mirrors the per-profile
  // pager): fetch that source's next window and merge it back in place, leaving
  // every other platform's rows untouched. Resolves the platform's exact total.
  const loadMoreMessagingForPlatform = useCallback(async (platform: string) => {
    const inPlatform = (s: SessionInfo) => normalizeSessionSource(s.source) === platform
    const loaded = $messagingSessions.get().filter(inPlatform).length

    const result = await listAllProfileSessions(loaded + SIDEBAR_SESSIONS_PAGE_SIZE, 1, 'exclude', 'recent', 'all', {
      source: platform
    })

    const incoming = result.sessions.filter(s => normalizeSessionSource(s.source) === platform)

    setMessagingSessions(prev => [
      ...prev.filter(s => !inPlatform(s)),
      ...mergeSessionPage(prev.filter(inPlatform), incoming, sessionsToKeep())
    ])

    const total = result.total ?? incoming.length
    setMessagingPlatformTotals(prev => ({ ...prev, [platform]: Math.max(total, incoming.length) }))
  }, [])

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
      // Unified cross-profile list (served read-only off each profile's
      // state.db; no per-profile backend is spawned). Single-profile users get
      // the same rows tagged profile="default".
      // Scope recents to the active profile (not always 'all') so a profile
      // with few recent sessions isn't windowed out of the cross-profile
      // recency page — the empty-history-on-profile-switch bug. Cron + messaging
      // stay cross-profile.
      const sessionProfile = profileScope === ALL_PROFILES ? 'all' : profileScope

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

      if (refreshSessionsRequestRef.current === requestId) {
        const recents = result.recents

        // Drop rows the user just deleted/archived: a refresh can race an
        // in-flight mutation and the backend page still carries the doomed row.
        // Honoring the optimistic tombstone keeps the removal from flashing back
        // (the tombstone self-clears once projects.tree confirms the delete).
        const tombstones = $removedSessionIds.get()

        const incoming = tombstones.size
          ? recents.sessions.filter(
              s => !tombstones.has(s.id) && !(s._lineage_root_id && tombstones.has(s._lineage_root_id))
            )
          : recents.sessions

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

        // Cron section: latest N cron sessions (kept so a pinned cron run still
        // resolves via sessionByAnyId), signature-gated like above.
        setCronSessions(prev => (sameCronSignature(prev, result.cron.sessions) ? prev : result.cron.sessions))

        // Messaging sections: drop any non-messaging source the broad exclude
        // didn't catch (custom sources stay in local recents), then split per
        // platform in the UI.
        const messagingRows = result.messaging.sessions.filter(s => isMessagingSource(s.source))

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
  }, [profileScope, refreshCronJobs])

  const loadMoreSessions = useCallback(async () => {
    bumpSessionsLimit()
    await refreshSessions()
  }, [refreshSessions])

  // Permanently delete every chat in the active profile scope - the same rows
  // the recents list shows (non-archived; cron/messaging/subagent excluded).
  // Pages the scope in <=500-id chunks and deletes each via the bulk endpoint
  // until it's empty, so it clears the whole history rather than only the
  // currently-loaded window, with no new backend route. Archived chats live in
  // Settings and are intentionally left untouched. Returns the number removed.
  const clearAllSessions = useCallback(async (): Promise<number> => {
    const sessionProfile = profileScope === ALL_PROFILES ? 'all' : profileScope

    let removed = 0

    try {
      for (let page = 0; page < SESSION_CLEAR_MAX_PAGES; page++) {
        const result = await listAllProfileSessions(BULK_DELETE_MAX_IDS, 1, 'exclude', 'recent', sessionProfile, {
          excludeSources: SIDEBAR_EXCLUDED_SOURCES
        })

        if (result.sessions.length === 0) {
          break
        }

        // Group by owning profile: each profile has its own state.db and the
        // endpoint scopes to one profile per call, mirroring the single-session
        // delete which routes by the row's own `profile`.
        const idsByProfile = new Map<string, string[]>()

        for (const session of result.sessions) {
          const key = session.profile ?? 'default'
          const ids = idsByProfile.get(key)

          if (ids) {
            ids.push(session.id)
          } else {
            idsByProfile.set(key, [session.id])
          }
        }

        let deletedThisPage = 0

        for (const [profile, ids] of idsByProfile) {
          const { deleted } = await bulkDeleteSessions(ids, profile)
          deletedThisPage += deleted

          // Drop the rows + their pins optimistically so the sidebar empties as we
          // page rather than snapping clear only at the closing refresh.
          const goneIds = new Set(ids)
          const gonePins = new Set(result.sessions.filter(s => goneIds.has(s.id)).map(sessionPinId))
          setSessions(prev => prev.filter(s => !goneIds.has(s.id)))
          $pinnedSessionIds.set($pinnedSessionIds.get().filter(id => !goneIds.has(id) && !gonePins.has(id)))
        }

        removed += deletedThisPage

        // Nothing in this page actually deleted (every id was already gone, or the
        // backend declined) - bail instead of re-fetching the same page forever.
        if (deletedThisPage === 0) {
          break
        }
      }
    } finally {
      // Re-pull the authoritative list so totals/footer - and any chat created
      // mid-clear - are accurate. In the finally so a batch that rejects midway
      // still reconciles the optimistic removals already applied above (the
      // rejection itself keeps propagating to the caller).
      await refreshSessions()
    }

    return removed
  }, [profileScope, refreshSessions])

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

    // A full window back means the profile still has more on disk.
    const truncated = result.sessions.length >= loaded + SIDEBAR_SESSIONS_PAGE_SIZE
    setSessionProfilesTruncated(prev => ({ ...prev, [key]: truncated }))
  }, [])

  return {
    clearAllSessions,
    loadMoreMessagingForPlatform,
    loadMoreSessions,
    loadMoreSessionsForProfile,
    refreshCronJobs,
    refreshMessagingSessions,
    refreshSessions
  }
}
