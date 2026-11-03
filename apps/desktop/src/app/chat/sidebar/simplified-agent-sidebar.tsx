import { useStore } from '@nanostores/react'
import { type ComponentProps, useMemo, useState } from 'react'

import { ambientOwnerConnectionId } from '@/api/client'
import { mutateSessionHttp } from '@/api/sessions'
import { SidebarPanelLabel } from '@/app/shell/sidebar-label'
import { Codicon } from '@/components/ui/codicon'
import { RowButton } from '@/components/ui/row-button'
import { SearchField } from '@/components/ui/search-field'
import { Sidebar, SidebarContent } from '@/components/ui/sidebar'
import { type SessionInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionMatchesSearch } from '@/lib/session-search'
import { $pinnedSessionIds, $sidebarPinsOpen, setSidebarPinsOpen } from '@/store/layout'
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
  $sessionsLoading
} from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $unconfirmedPinWrites } from '@/store/session-pin-sync'
import { $removedSessionIds } from '@/store/session-removal'

import { SidebarLoadMoreRow } from './load-more-row'
import { SidebarLoadErrorState, SidebarSessionSkeletons, SidebarStorageCorruptNotice } from './section-states'
import { SimplifiedAgentList } from './simplified-agent-list'
import { SimplifiedNewSession } from './simplified-new-session'
import { sessionRowKey } from './simplified-session-identity'
import { SimplifiedSessionList } from './simplified-session-list'
import { ambiguousSessionIds, simplifiedPins } from './simplified-session-model'
import { useSimplifiedSessionSearch } from './use-simplified-session-search'

import type { ChatSidebar } from './index'

export function SimplifiedAgentSidebar(props: ComponentProps<typeof ChatSidebar>) {
  const { t } = useI18n()
  const sessions = useStore($sessions)
  const messaging = useStore($messagingSessions)
  const cron = useStore($cronSessions)
  const removed = useStore($removedSessionIds)
  const pinIds = useStore($pinnedSessionIds)
  const unconfirmed = useStore($unconfirmedPinWrites)
  const pinsOpen = useStore($sidebarPinsOpen)
  const activeSessionId = useStore($focusedStoredSessionId)
  const loading = useStore($sessionsLoading)
  const error = useStore($sessionsLoadError)
  const truncated = useStore($sessionProfilesTruncated)
  const messagingTruncated = useStore($messagingTruncated)
  const messagingTotals = useStore($messagingPlatformTotals)
  const [query, setQuery] = useState('')
  const [loadingMore, setLoadingMore] = useState(false)

  const all = useMemo(
    () =>
      [...new Map([...cron, ...messaging, ...sessions].map(row => [sessionRowKey(row), row])).values()]
        .filter(row => !row.hidden && !row.archived && !removed.has(row.id) && !removed.has(sessionRowKey(row)))
        .sort((a, b) => (b.last_active ?? b.started_at) - (a.last_active ?? a.started_at)),
    [sessions, messaging, cron, removed]
  )

  const pinned = useMemo(() => simplifiedPins(all, pinIds, unconfirmed), [all, pinIds, unconfirmed])

  const pinnedIds = new Set(pinned.map(row => sessionRowKey(row)))

  const matches = (row: (typeof all)[number]) =>
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

  const shared = {
    activeSessionId,
    pinnedIds,
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
      <SidebarContent className="gap-2.5 overflow-hidden px-3 pb-3 pt-[calc(var(--titlebar-height)+0.5rem)]">
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
        <section className="max-h-[25%] shrink-0 overflow-y-auto">
          <RowButton
            aria-expanded={pinsOpen}
            className="flex w-full items-center rounded-sm text-left hover:bg-(--chrome-action-hover) focus-visible:outline-2 focus-visible:outline-ring"
            onClick={() => setSidebarPinsOpen(!pinsOpen)}
          >
            <SidebarPanelLabel className="flex-1 px-2 py-2 [&>span:first-child]:mx-1">
              {t.sidebar.pinned}
            </SidebarPanelLabel>
            <Codicon
              className="mr-2 text-(--ui-text-tertiary)"
              name={pinsOpen ? 'chevron-down' : 'chevron-right'}
              size={12}
            />
          </RowButton>
          {pinsOpen && <SimplifiedSessionList {...shared} sessions={pinned.filter(matches)} showAgent />}
        </section>
        <SimplifiedAgentList
          renderSessions={agentSessions => (
            <div className="mb-2 ml-8 mt-0.5 border-l border-(--ui-stroke-secondary) pl-2">
              <SimplifiedSessionList {...shared} sessions={agentSessions} />
            </div>
          )}
          searching={Boolean(query.trim())}
          sessions={rows}
        />
        {loading && rows.length === 0 && <SidebarSessionSkeletons />}
        {!loading && rows.length === 0 && (
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
