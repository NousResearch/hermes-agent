import { useStore } from '@nanostores/react'
import { type ComponentProps, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { SearchField } from '@/components/ui/search-field'
import { Sidebar, SidebarContent } from '@/components/ui/sidebar'
import { useI18n } from '@/i18n'
import { sessionMatchesSearch } from '@/lib/session-search'
import { $pinnedSessionIds, $sidebarPinsOpen, pinSession, setSidebarPinsOpen, unpinSession } from '@/store/layout'
import { notifyError } from '@/store/notifications'
import { $newChatProfile, ALL_PROFILES } from '@/store/profile'
import {
  $cronSessions,
  $messagingSessions,
  $sessionProfilesTruncated,
  $sessions,
  $sessionsLoadError,
  $sessionsLoading
} from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $unconfirmedPinWrites } from '@/store/session-pin-sync'
import { $removedSessionIds } from '@/store/session-removal'
import { markSessionUnread } from '@/store/session-unread-remote'

import { SidebarLoadMoreRow } from './load-more-row'
import { SidebarLoadErrorState, SidebarSessionSkeletons, SidebarStorageCorruptNotice } from './section-states'
import { buildSessionByAnyId, resolvePinnedSessions } from './session-index'
import { SidebarSessionsSection, VIRTUALIZE_THRESHOLD } from './sessions-section'
import { SimplifiedAgentList } from './simplified-agent-list'
import { useServerSessionSearch } from './use-server-session-search'

import { type ChatSidebar, mergeSearchResults } from './index'

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
  const [query, setQuery] = useState('')
  const [loadingMore, setLoadingMore] = useState(false)

  const all = useMemo(
    () =>
      [...new Map([...cron, ...messaging, ...sessions].map(row => [row.id, row])).values()]
        .filter(row => !row.hidden && !row.archived && !removed.has(row.id))
        .sort((a, b) => (b.last_active ?? b.started_at) - (a.last_active ?? a.started_at)),
    [sessions, messaging, cron, removed]
  )

  const pinned = useMemo(
    () => resolvePinnedSessions(pinIds, buildSessionByAnyId(all, [], []), all, unconfirmed),
    [all, pinIds, unconfirmed]
  )

  const pinnedIds = new Set(pinned.map(row => row.id))

  const matches = (row: (typeof all)[number]) =>
    !query.trim() ||
    sessionMatchesSearch(row, query.trim()) ||
    (row.profile ?? 'default').toLowerCase().includes(query.trim().toLowerCase())

  const { serverMatches, searchPending } = useServerSessionSearch(query.trim(), ALL_PROFILES)

  const searchRows = query.trim()
    ? mergeSearchResults(all, query.trim(), serverMatches, buildSessionByAnyId(all, [], []), searchPending)
    : all

  const rows = searchRows.filter(row => !pinnedIds.has(row.id) && !row.hidden && !row.archived && !removed.has(row.id))

  const shared = {
    activeSessionId,
    onResumeSession: props.onResumeSession,
    onDeleteSession: props.onDeleteSession,
    onArchiveSession: props.onArchiveSession,
    onBranchSession: props.onBranchSession,
    onToggleUnread: (id: string) => {
      const row = all.find(session => session.id === id)

      if (row) {
        void markSessionUnread(id, row.unread !== true).catch(error => notifyError(error, t.sidebar.row.unreadFailed))
      }
    },
    showProfileTags: true,
    grouping: 'none' as const,
    preserveOrder: true,
    contentClassName: 'flex flex-col gap-px',
    rootClassName: 'shrink-0 p-0'
  }

  async function loadMore() {
    setLoadingMore(true)

    try {
      await props.onLoadMoreSessions()
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
      <SidebarContent className="gap-2 overflow-hidden px-2.5 pb-2 pt-[calc(var(--titlebar-height)+0.375rem)]">
        <Button
          className="w-full justify-start"
          onClick={() => {
            $newChatProfile.set(null)
            props.onNavigate({
              id: 'new-session',
              label: t.sidebar.nav['new-session'],
              action: 'new-session',
              icon: () => null,
              keybindActionId: 'session.new'
            })
          }}
          variant="ghost"
        >
          <Codicon name="add" />
          {t.sidebar.nav['new-session']}
        </Button>
        <SearchField
          aria-label={t.sidebar.searchAria}
          onChange={setQuery}
          placeholder={t.sidebar.searchPlaceholder}
          value={query}
        />
        <SidebarStorageCorruptNotice />
        <SidebarSessionsSection
          {...shared}
          emptyState={null}
          label={t.sidebar.pinned}
          onToggle={() => setSidebarPinsOpen(!pinsOpen)}
          onTogglePin={unpinSession}
          open={pinsOpen}
          pinned
          rootClassName="max-h-[25%] shrink-0 overflow-y-auto p-0"
          sessions={pinned.filter(matches)}
        />
        <SimplifiedAgentList
          sessions={rows}
          renderSessions={agentSessions => (
            <SidebarSessionsSection
              {...shared}
              contentClassName="flex min-h-0 flex-1 flex-col"
              emptyState={null}
              onToggle={() => {}}
              onTogglePin={pinSession}
              open
              pinned={false}
              rootClassName={
                agentSessions.length >= VIRTUALIZE_THRESHOLD
                  ? 'h-64 min-h-0 overflow-hidden py-0 pr-0 pl-3'
                  : 'py-0 pr-0 pl-3'
              }
              sessions={agentSessions}
            />
          )}
        />
        {loading && rows.length === 0 && <SidebarSessionSkeletons />}
        {!loading && rows.length === 0 && (
          <p className="px-2 py-3 text-xs text-(--ui-text-tertiary)">
            {query ? t.sidebar.noMatch(query) : t.sidebar.noSessions}
          </p>
        )}
        {Object.values(truncated).some(Boolean) && (
          <SidebarLoadMoreRow loading={loading || loadingMore} onClick={() => void loadMore()} step={0} />
        )}
        {error && <SidebarLoadErrorState onRetry={() => void props.onRetrySessions()} />}
      </SidebarContent>
    </Sidebar>
  )
}
