import { useStore } from '@nanostores/react'
import { type ComponentProps, type SetStateAction, useMemo, useState } from 'react'

import { mutateSessionHttp } from '@/api/sessions'
import { type SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionMatchesSearch } from '@/lib/session-search'
import { MESSAGING_SESSION_SOURCE_IDS, normalizeSessionSource } from '@/lib/session-source'
import { $hiddenSidebarAgents, $sidebarAgentExpanded, agentSidebarKey } from '@/store/agent-sidebar-preferences'
import { $pinnedSessionIds } from '@/store/layout'
import { notifyError } from '@/store/notifications'
import { $sessionListProfileScope, messagingTotalsKey, sidebarProfileForScope } from '@/store/profile'
import {
  $cronSessions,
  $messagingPlatformTotals,
  $messagingSessions,
  $messagingTruncated,
  $sessionProfilesTruncated,
  $sessions,
  $sessionsLoadError,
  $sessionsLoading,
  sessionPinId
} from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $unconfirmedPinWrites } from '@/store/session-pin-sync'
import { $removedSessionIds } from '@/store/session-removal'

import { sessionRowKey } from './simplified-session-identity'
import { ambiguousSessionIds, simplifiedPins } from './simplified-session-model'
import { useIaSearchMetrics } from './use-ia-search-metrics'
import { useSimplifiedAgents, useSimplifiedOwner } from './use-simplified-agents'
import { useSimplifiedPinWrites } from './use-simplified-pin-writes'
import { useSimplifiedSessionSearch } from './use-simplified-session-search'

import type { ChatSidebar } from './index'

/** Session ownership and writes for the normal agent sidebar. */
export function useAgentSidebar(props: ComponentProps<typeof ChatSidebar>) {
  const { t } = useI18n()
  const owner = useSimplifiedOwner()
  const pinWrites = useSimplifiedPinWrites(owner, props.onRetrySessions)
  const sessions = useStore($sessions)
  const messaging = useStore($messagingSessions)
  const cron = useStore($cronSessions)
  const removed = useStore($removedSessionIds)
  const pinIds = useStore($pinnedSessionIds)
  const unconfirmed = useStore($unconfirmedPinWrites)
  const activeSessionId = useStore($focusedStoredSessionId)
  const loading = useStore($sessionsLoading)
  const error = useStore($sessionsLoadError)
  const truncated = useStore($sessionProfilesTruncated)
  const messagingTruncated = useStore($messagingTruncated)
  const messagingTotals = useStore($messagingPlatformTotals)
  const messagingProfile = sidebarProfileForScope(useStore($sessionListProfileScope))

  const pageableMessaging = useMemo(
    () =>
      MESSAGING_SESSION_SOURCE_IDS.filter(platform => {
        const loaded = messaging.filter(
          row =>
            normalizeSessionSource(row.source) === platform &&
            (messagingProfile === 'all' || (row.profile || 'default') === messagingProfile)
        ).length

        const total = messagingTotals[messagingTotalsKey(messagingProfile, platform)]

        return loaded > 0 && (total === undefined ? messagingTruncated : loaded < total)
      }),
    [messaging, messagingProfile, messagingTotals, messagingTruncated]
  )

  const [query, setQuery] = useState('')
  const expanded = useStore($sidebarAgentExpanded)
  const hiddenAgents = useStore($hiddenSidebarAgents)

  const setExpanded = (value: SetStateAction<Record<string, boolean>>) => {
    $sidebarAgentExpanded.set(typeof value === 'function' ? value($sidebarAgentExpanded.get()) : value)
  }

  const [revealSession, setRevealSession] = useState<{ key: string } | null>(null)
  const [loadingMore, setLoadingMore] = useState(false)

  const all = useMemo(
    () =>
      [
        ...new Map(
          [...pinWrites.rows, ...cron, ...messaging, ...sessions].map(row => [
            sessionRowKey(row, sessionPinId(row)),
            row
          ])
        ).values()
      ]
        .filter(row => !row.hidden && !row.archived && !removed.has(row.id) && !removed.has(sessionRowKey(row)))
        .sort((a, b) => (b.last_active ?? b.started_at) - (a.last_active ?? a.started_at)),
    [sessions, messaging, cron, removed, pinWrites.rows]
  )

  const matches = (row: SessionInfo) =>
    !query.trim() ||
    sessionMatchesSearch(row, query.trim()) ||
    (row.profile ?? 'default').toLowerCase().includes(query.trim().toLowerCase())

  const search = useSimplifiedSessionSearch(query.trim())

  const searchRows = query.trim()
    ? [...new Map([...search.rows, ...all.filter(matches)].map(row => [sessionRowKey(row), row])).values()]
    : all

  const rows = searchRows.filter(
    row => !row.hidden && !row.archived && !removed.has(row.id) && !removed.has(sessionRowKey(row))
  )

  useIaSearchMetrics(query.trim(), search.pending, search.failed, rows.length)
  const agents = useSimplifiedAgents(rows)
  const pinned = simplifiedPins(all, pinIds, unconfirmed, pinWrites.value)
  const pinnedIds = new Set(pinned.map(row => sessionRowKey(row)))

  const shared = {
    searching: Boolean(query.trim()),
    activeSessionId,
    pinnedIds,
    pinPending: pinWrites.pending,
    onResumeSession: props.onResumeSession,
    onDeleteSession: props.onDeleteSession,
    onArchiveSession: props.onArchiveSession,
    onBranchSession: props.onBranchSession,
    ambiguousIds: ambiguousSessionIds([...all, ...rows]),
    onUpdate: (row: SessionInfo, field: 'pinned' | 'unread', value: boolean) => {
      void updateRow(row, field, value)
    }
  }

  async function updateRow(row: SessionInfo, field: 'pinned' | 'unread', value: boolean) {
    const scope = { connectionId: row.connection_id || undefined, profile: row.profile || 'default' }

    try {
      if (field === 'pinned') {
        await pinWrites.setPinned(row, value)

        return
      }

      await mutateSessionHttp(row.id, 'PATCH', { [field]: value }, scope)
      await props.onRetrySessions()
    } catch (error) {
      notifyError(error, t.errors.genericFailure)
    }
  }

  async function loadMore() {
    setLoadingMore(true)

    try {
      await Promise.all([
        props.onLoadMoreSessions(),
        ...(props.onLoadMoreMessaging ? pageableMessaging.map(platform => props.onLoadMoreMessaging!(platform)) : [])
      ])
    } catch (error) {
      notifyError(error, t.errors.genericFailure)
    } finally {
      setLoadingMore(false)
    }
  }

  function resumePinned(id: string, row?: SessionInfo) {
    if (row) {
      setRevealSession({ key: sessionRowKey(row) })
      const owner = row.connection_id || agents.owner
      setExpanded(current => ({
        ...current,
        [JSON.stringify(['gateway', owner])]: true,
        [agentSidebarKey({ connectionId: owner, profile: row.profile || 'default' })]: true
      }))
    }

    props.onResumeSession(id, row)
  }

  return {
    t,
    all,
    query,
    setQuery,
    search,
    rows,
    shared,
    agents,
    expanded,
    setExpanded,
    revealSession,
    loading,
    error,
    loadingMore,
    loadMore,
    resumePinned,
    pinned: pinned.filter(
      row =>
        matches(row) &&
        (query.trim() ||
          !hiddenAgents.includes(
            agentSidebarKey({
              connectionId: row.connection_id || owner,
              profile: row.profile || 'default'
            })
          ))
    ),
    hasMore: Object.values(truncated).some(Boolean) || Boolean(pageableMessaging.length && props.onLoadMoreMessaging)
  }
}
