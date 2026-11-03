import { useStore } from '@nanostores/react'
import { type ComponentProps, useMemo, useState } from 'react'

import { ambientOwnerConnectionId } from '@/api/client'
import { mutateSessionHttp } from '@/api/sessions'
import { SearchField } from '@/components/ui/search-field'
import { Sidebar, SidebarContent } from '@/components/ui/sidebar'
import { type SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionMatchesSearch } from '@/lib/session-search'
import { $pinnedSessionIds } from '@/store/layout'
import { notifyError } from '@/store/notifications'
import { $newChatProfile } from '@/store/profile'
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

import { SidebarLoadMoreRow } from './load-more-row'
import { SidebarLoadErrorState, SidebarSessionSkeletons, SidebarStorageCorruptNotice } from './section-states'
import { SimplifiedAgentList } from './simplified-agent-list'
import { SimplifiedNewSession } from './simplified-new-session'
import { SimplifiedPinnedSessions } from './simplified-pinned-sessions'
import { sessionRowKey } from './simplified-session-identity'
import { SimplifiedSessionList } from './simplified-session-list'
import { ambiguousSessionIds, simplifiedPins } from './simplified-session-model'
import { useSimplifiedAgents, useSimplifiedOwner } from './use-simplified-agents'
import { useSimplifiedPinWrites } from './use-simplified-pin-writes'
import { useSimplifiedSessionSearch } from './use-simplified-session-search'

import type { ChatSidebar } from './index'

export function SimplifiedAgentSidebar(props: ComponentProps<typeof ChatSidebar>) {
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
  const [query, setQuery] = useState('')
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [loadingMore, setLoadingMore] = useState(false)

  const all = useMemo(
    () =>
      [
        ...new Map(
          [...pinWrites.rows, ...cron, ...messaging, ...sessions]
            .map(row => ({ ...row, connection_id: row.connection_id || owner || undefined }))
            .map(row => [sessionRowKey(row, sessionPinId(row)), row])
        ).values()
      ]
        .filter(row => !row.hidden && !row.archived && !removed.has(row.id) && !removed.has(sessionRowKey(row)))
        .sort((a, b) => (b.last_active ?? b.started_at) - (a.last_active ?? a.started_at)),
    [sessions, messaging, cron, removed, pinWrites.rows, owner]
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

  const agents = useSimplifiedAgents(rows)
  const pinned = simplifiedPins(all, pinIds, unconfirmed, pinWrites.value)
  const pinnedIds = new Set(pinned.map(row => sessionRowKey(row)))

  const shared = {
    activeSessionId,
    pinnedIds,
    pinPending: pinWrites.pending,
    onResumeSession: props.onResumeSession,
    onDeleteSession: props.onDeleteSession,
    onArchiveSession: props.onArchiveSession,
    onBranchSession: props.onBranchSession,
    ambiguousIds: ambiguousSessionIds(all),
    onUpdate: (row: SessionInfo, field: 'pinned' | 'unread', value: boolean) => {
      void updateRow(row, field, value)
    }
  }

  async function updateRow(row: SessionInfo, field: 'pinned' | 'unread', value: boolean) {
    const scope = { connectionId: row.connection_id || ambientOwnerConnectionId(), profile: row.profile || 'default' }

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
        ...(messagingTruncated && props.onLoadMoreMessaging
          ? Object.keys(messagingTotals).map(platform => props.onLoadMoreMessaging!(platform))
          : [])
      ])
    } catch (error) {
      notifyError(error, t.errors.genericFailure)
    } finally {
      setLoadingMore(false)
    }
  }

  return (
    <Sidebar
      className="relative h-full min-w-0 overflow-hidden border-r border-(--sidebar-edge-border) bg-(--ui-sidebar-surface-background) text-foreground"
      collapsible="none"
      data-tour="sessions-sidebar"
    >
      <SidebarContent className="gap-2.5 overflow-y-auto px-3 pb-3 pt-[calc(var(--titlebar-height)+0.5rem)]">
        <SimplifiedNewSession
          onFallback={() => {
            $newChatProfile.set(null)
            props.onNavigate({
              id: 'new-session',
              label: t.sidebar.nav['new-session'],
              action: 'new-session',
              icon: () => null,
              keybindActionId: 'session.new'
            })
          }}
          sessions={all}
        />
        <SearchField
          aria-label={t.sidebar.searchAria}
          containerClassName="mx-0 gap-2.5 px-2 opacity-60 focus-within:opacity-100"
          loading={search.pending}
          onChange={setQuery}
          placeholder={t.sidebar.searchPlaceholder}
          value={query}
        />
        <SidebarStorageCorruptNotice />
        {search.failed && (
          <p className="px-2 text-xs text-destructive" role="alert">
            {t.errors.genericFailure}
          </p>
        )}
        <div className="flex min-h-44 flex-1 flex-col gap-2.5">
          <SimplifiedPinnedSessions
            {...shared}
            onResumeSession={(id, row) => {
              if (row) {
                const owner = row.connection_id || agents.owner
                setExpanded(current => ({
                  ...current,
                  [JSON.stringify(['gateway', owner])]: true,
                  [JSON.stringify([owner, row.profile || 'default'])]: true
                }))
              }

              props.onResumeSession(id, row)
            }}
            sessions={pinned.filter(matches)}
          />
          <SimplifiedAgentList
            agents={agents}
            expanded={expanded}
            onToggle={(id, open) => setExpanded(current => ({ ...current, [id]: !open }))}
            renderSessions={agentSessions => (
              <div className="mb-2 ml-8 mt-0.5 border-l border-(--ui-stroke-secondary) pl-2">
                <SimplifiedSessionList {...shared} sessions={agentSessions} />
              </div>
            )}
            searching={Boolean(query.trim())}
          />
        </div>
        {loading && rows.length === 0 && <SidebarSessionSkeletons />}
        {query.trim() && search.pending && (
          <p className="sr-only" role="status">
            {t.common.loading}
          </p>
        )}
        {!loading && !search.pending && !search.failed && rows.length === 0 && (
          <p className="px-2 py-3 text-xs text-(--ui-text-tertiary)">
            {query ? t.sidebar.noMatch(query) : t.sidebar.noSessions}
          </p>
        )}
        {(Object.values(truncated).some(Boolean) || (messagingTruncated && props.onLoadMoreMessaging)) && (
          <SidebarLoadMoreRow loading={loading || loadingMore} onClick={() => void loadMore()} step={0} />
        )}
        {error && <SidebarLoadErrorState onRetry={() => void props.onRetrySessions()} />}
      </SidebarContent>
    </Sidebar>
  )
}
