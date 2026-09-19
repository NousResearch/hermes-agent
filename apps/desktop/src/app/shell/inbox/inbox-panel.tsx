import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import {
  Panel,
  PanelBody,
  PanelEmpty,
  PanelHeader,
  PanelListRow,
  PanelMeta,
  PanelSectionLabel
} from '@/app/overlays/panel'
import { sessionRoute } from '@/app/routes'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { SearchField } from '@/components/ui/search-field'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { cn } from '@/lib/utils'
import {
  countByCategory,
  dismissExpiredRequest,
  fetchInboxRequestDetails,
  filterNeedsAttention,
  type InboxCategory,
  type InboxEntry,
  type InboxExpiredRequest,
  type InboxItem,
  type InboxRequestContext,
  type InboxRequestDetails,
  redoExpiredRequest,
  refreshInbox,
  searchInboxItems
} from '@/store/inbox'
import { $gateway } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, setSelectedStoredSessionId } from '@/store/session'

import { ApprovalCard } from './approval-card'
import { AutomationControls } from './automation-controls'
import { ClarifyCard } from './clarify-card'

type SearchScope = 'all' | 'section'

const EXPIRY_OUTCOME_TEXT: Record<string, string> = {
  notify_failed: 'never reached a surface that could answer it',
  session_closed: 'the session closed before an answer',
  timeout: 'timed out without an answer'
}

function expiryLine(entry: InboxExpiredRequest): string {
  const seconds = Math.max(0, Date.now() / 1000 - entry.ended_at)
  const ago =
    seconds < 90 ? 'just now'
      : seconds < 3600 ? `${Math.round(seconds / 60)}m ago`
        : seconds < 86_400 ? `${Math.round(seconds / 3600)}h ago`
          : `${Math.round(seconds / 86_400)}d ago`

  return `Expired ${ago} — ${EXPIRY_OUTCOME_TEXT[entry.outcome] ?? entry.outcome}`
}

/**
 * One request that ended without an answer. Kept visible instead of vanishing: what it was
 * for, that it expired, a Redo that re-raises it through the session, and a Dismiss for
 * the operator's "I'm done with this one".
 */
function ExpiredRequestCard({
  entry,
  onChanged,
  sessionKey
}: {
  entry: InboxExpiredRequest
  onChanged: () => void
  sessionKey: string
}) {
  const [busy, setBusy] = useState<'dismiss' | 'redo' | null>(null)
  const [error, setError] = useState<string | null>(null)

  const act = async (kind: 'dismiss' | 'redo') => {
    setBusy(kind)
    setError(null)

    try {
      if (kind === 'redo') {
        await redoExpiredRequest({ requestId: entry.request_id, sessionKey })
      } else {
        await dismissExpiredRequest({ requestId: entry.request_id, sessionKey })
      }

      onChanged()
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'request failed')
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="rounded-md border border-(--ui-stroke-tertiary) bg-foreground/5 px-2 py-1.5" data-expired-request="">
      <p className="break-words font-mono text-[0.66rem] text-foreground/85">{entry.command || '(command not recorded)'}</p>
      {entry.description && (
        <p className="mt-0.5 break-words text-[0.62rem] text-muted-foreground/70">{entry.description}</p>
      )}
      <p className="mt-1 text-[0.6rem] text-muted-foreground/60">{expiryLine(entry)}</p>
      <div className="mt-1.5 flex gap-2">
        <Button disabled={busy !== null} onClick={() => void act('redo')} size="xs" variant="secondary">
          {busy === 'redo' ? 'Redoing…' : 'Redo'}
        </Button>
        <Button disabled={busy !== null} onClick={() => void act('dismiss')} size="xs" variant="text">
          {busy === 'dismiss' ? 'Dismissing…' : 'Dismiss'}
        </Button>
      </div>
      {error && (
        <p className="mt-1 break-words text-[0.62rem] text-destructive" role="alert">
          {error}
        </p>
      )}
    </div>
  )
}

interface InlineDetailProps {
  detailsLoading: boolean | undefined
  detailsError: string | null | undefined
  expandedDetails: InboxRequestDetails | null | undefined
  item: InboxItem
  onOpenSession: (item: InboxItem) => void
  onRetry: () => void
}

function InlineDetail({ detailsLoading, detailsError, expandedDetails, item, onOpenSession, onRetry }: InlineDetailProps) {
  const detailApprovals = expandedDetails?.sessions.flatMap(s => s.approvals) ?? []
  const detailClarifications = expandedDetails?.sessions.flatMap(s => s.clarifications) ?? []
  const detailLiveSessionId = expandedDetails?.sessions[0]?.live_session_ids[0] ?? ''

  const detailContext: InboxRequestContext = expandedDetails?.sessions[0]?.context ?? {
    available: false,
    messages: [],
    reason: null
  }
  const detailExpired = expandedDetails?.sessions[0]?.expired_requests ?? []

  return (
    <div className="flex flex-col gap-3 py-2">
      <PanelMeta
        rows={[
          { label: 'Title', value: item.title || '—' },
          { label: 'Session', value: item.session_key },
          { label: 'Lanes', value: item.lanes.map(lane => LANE_LABEL[lane] ?? lane).join(', ') },
          { label: 'Source', value: item.source || '—' },
          { label: 'Cwd', value: item.cwd || '—' },
          ...(item.subagent_count > 0 || item.subagent_count_unavailable
            ? [{ label: 'Subagents', value: item.subagent_count_unavailable ? `${item.subagent_count}+ (unavailable)` : String(item.subagent_count) }]
            : []),
          ...(item.background_task_count > 0 || item.background_task_count_unavailable
            ? [{ label: 'BG tasks', value: item.background_task_count_unavailable ? `${item.background_task_count}+ (unavailable)` : String(item.background_task_count) }]
            : []),
          ...(item.categories.length > 0
            ? [{ label: 'Categories', value: item.categories.join(', ') }]
            : [])
        ]}
      />

      {item.goal && (
        <div>
          <PanelSectionLabel>Goal</PanelSectionLabel>
          <PanelMeta
            rows={[
              { label: 'Title', value: String((item.goal as Record<string, unknown>).title ?? '—') },
              { label: 'Status', value: String((item.goal as Record<string, unknown>).status ?? '—') }
            ]}
          />
          <AutomationControls
            kind="goal"
            liveSessionId={detailLiveSessionId}
            onChanged={onRetry}
            status={String((item.goal as Record<string, unknown>).status ?? '')}
          />
        </div>
      )}
      {item.loop && (
        <div>
          <PanelSectionLabel>Loop</PanelSectionLabel>
          <PanelMeta
            rows={[{ label: 'Status', value: String((item.loop as Record<string, unknown>).status ?? '—') }]}
          />
          <AutomationControls
            kind="loop"
            liveSessionId={detailLiveSessionId}
            onChanged={onRetry}
            status={String((item.loop as Record<string, unknown>).status ?? '')}
          />
        </div>
      )}
      {item.heartbeat && (
        <div>
          <PanelSectionLabel>Heartbeat</PanelSectionLabel>
          <PanelMeta
            rows={[{ label: 'Status', value: String((item.heartbeat as Record<string, unknown>).status ?? '—') }]}
          />
          <AutomationControls
            kind="heartbeat"
            liveSessionId={detailLiveSessionId}
            onChanged={onRetry}
            status={String((item.heartbeat as Record<string, unknown>).status ?? '')}
          />
        </div>
      )}

      {detailsLoading && <p role="status">Loading request details…</p>}
      {detailsError && (
        <div role="alert">
          <p className="text-sm text-destructive">{detailsError}</p>
          <Button onClick={onRetry} size="xs" variant="secondary">
            Retry request details
          </Button>
        </div>
      )}

      {/* Context first: decide from the conversation, then act — the point of the
          panel is to answer a request without leaving it. */}
      <div>
        <PanelSectionLabel>Recent messages</PanelSectionLabel>
        {detailContext.available && detailContext.messages.length > 0 ? (
          <ul className="mt-1 flex flex-col gap-1.5">
            {detailContext.messages.map((message, index) => (
              <li
                className="rounded-md border border-(--ui-stroke-tertiary) bg-foreground/5 px-2 py-1.5"
                key={`${message.role}-${message.timestamp ?? index}-${index}`}
              >
                <span className="text-[0.6rem] uppercase tracking-wide text-muted-foreground/60">{message.role}</span>
                <p className="mt-0.5 whitespace-pre-wrap break-words text-[0.68rem] text-foreground/85">{message.text}</p>
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-1 text-[0.62rem] text-muted-foreground/60">
            {detailContext.reason ? `Unavailable: ${detailContext.reason}` : 'No recent transcript available.'}
          </p>
        )}
      </div>

      {detailApprovals.length > 0 && (
        <div>
          <PanelSectionLabel>Approvals ({detailApprovals.length})</PanelSectionLabel>
          <div className="mt-1 flex flex-col gap-2">
            {detailApprovals.map(a => (
              <ApprovalCard
                approval={a}
                key={a.request_id}
                liveSessionId={detailLiveSessionId}
                onResolved={onRetry}
              />
            ))}
          </div>
        </div>
      )}
      {detailClarifications.length > 0 && (
        <div>
          <PanelSectionLabel>Questions ({detailClarifications.length})</PanelSectionLabel>
          <div className="mt-1 flex flex-col gap-2">
            {detailClarifications.map(c => (
              <ClarifyCard
                clarification={c}
                key={c.request_id}
                onResolved={onRetry}
              />
            ))}
          </div>
        </div>
      )}

      {detailExpired.length > 0 && (
        <div>
          <PanelSectionLabel>Expired requests ({detailExpired.length})</PanelSectionLabel>
          <div className="mt-1 flex flex-col gap-2">
            {detailExpired.map(entry => (
              <ExpiredRequestCard
                entry={entry}
                key={entry.request_id}
                onChanged={onRetry}
                sessionKey={item.session_key}
              />
            ))}
          </div>
        </div>
      )}

      <div>
        <Button
          className="mt-1"
          onClick={() => onOpenSession(item)}
          size="xs"
          variant="secondary"
        >
          <Codicon name="arrow-right" size="0.75rem" />
          Open full chat
        </Button>
      </div>
    </div>
  )
}

const CATEGORY_NAV: { id: InboxCategory; label: string; icon: string }[] = [
  { id: 'all', label: 'All sessions', icon: 'inbox' },
  { id: 'goals', label: 'Goals', icon: 'target' },
  { id: 'loops', label: 'Loops', icon: 'sync' },
  { id: 'heartbeats', label: 'Heartbeats', icon: 'pulse' },
  { id: 'background_tasks', label: 'Background tasks', icon: 'tools' },
  { id: 'subagents', label: 'Subagents', icon: 'account' },
  { id: 'other', label: 'Other', icon: 'folder' }
]

const LANE_LABEL: Record<string, string> = {
  needs_you: 'Needs you',
  running: 'Running',
  waiting: 'Waiting',
  scheduled: 'Scheduled'
}

function laneDotClassName(item: InboxItem): string {
  if (item.lanes.includes('needs_you')) {return 'bg-destructive'}

  if (item.lanes.includes('running')) {return 'bg-primary/70'}

  if (item.lanes.includes('waiting')) {return 'bg-amber-500/70'}

  if (item.lanes.includes('scheduled')) {return 'bg-muted-foreground/40'}

  return 'bg-muted-foreground/30'
}

function formatCounts(item: InboxItem): string {
  const parts: string[] = []

  if (item.subagent_count > 0) {
    parts.push(`${item.subagent_count} ${item.subagent_count === 1 ? 'subagent' : 'subagents'}`)
  }

  if (item.background_task_count > 0) {
    parts.push(`${item.background_task_count} ${item.background_task_count === 1 ? 'background task' : 'background tasks'}`)
  }

  if (item.subagent_count_unavailable) {parts.push('Subagents unavailable')}

  if (item.background_task_count_unavailable) {parts.push('Background tasks unavailable')}

  return parts.join(' · ')
}

function metaLine(item: InboxItem): string {
  const lane = item.lanes.map(l => LANE_LABEL[l] ?? l).join(' · ')
  const counts = formatCounts(item)

  return counts ? `${lane} · ${counts}` : lane
}

export function InboxPanel({ inbox, onClose }: { inbox: InboxEntry; onClose: () => void }) {
  const navigate = useNavigate()
  const [category, setCategory] = useState<InboxCategory>('all')
  const [needsAttentionOnly, setNeedsAttentionOnly] = useState(false)
  const [query, setQuery] = useState('')
  const [searchScope, setSearchScope] = useState<SearchScope>('section')
  const [expandedKey, setExpandedKey] = useState<string | null>(null)
  const [detailsCache, setDetailsCache] = useState<Record<string, InboxRequestDetails>>({})
  const [detailsLoading, setDetailsLoading] = useState<Record<string, boolean>>({})
  const [detailsError, setDetailsError] = useState<Record<string, string | null>>({})

  const profile = useStore($activeGatewayProfile) ?? ''
  const connection = useStore($connection)
  const gateway = useStore($gateway)
  const connectionId = connection?.connectionId ?? connection?.baseUrl ?? ''
  const scopeKey = `${connectionId}\0${profile}`
  const scopeRef = useRef(scopeKey)
  const gatewayRef = useRef(gateway)
  const loadSeqRef = useRef(0)
  const mountedRef = useRef(true)

  // Clear details cache when profile, connection, or gateway identity changes (synchronous guard)
  if (scopeRef.current !== scopeKey || gatewayRef.current !== gateway) {
    scopeRef.current = scopeKey
    gatewayRef.current = gateway
    ++loadSeqRef.current
    // Reset state synchronously during render to prevent one frame of stale data
    setDetailsCache({})
    setDetailsLoading({})
    setDetailsError({})
    setExpandedKey(null)
  }

  // Invalidate generation on unmount so in-flight fetches are dropped
  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      ++loadSeqRef.current
    }
  }, [])

  const snapshot = inbox.snapshot
  const coverage = snapshot?.coverage
  const items = useMemo(() => snapshot?.items ?? [], [snapshot?.items])

  const needsAttentionCount = useMemo(() => filterNeedsAttention(items).length, [items])

  const visibleItems = useMemo(() => {
    // Global search ("All sessions") searches ALL items regardless of attention/category filter.
    // The attention/category filters only apply to "This section" scope.
    if (searchScope === 'all') {
      return searchInboxItems(items, query, category, 'all')
    }

    let filtered = items

    if (needsAttentionOnly) {
      filtered = filterNeedsAttention(filtered)
    }

    return searchInboxItems(filtered, query, category, 'section')
  }, [items, query, category, searchScope, needsAttentionOnly])

  const loadDetails = useCallback(async (sessionKey: string) => {
    const seq = ++loadSeqRef.current
    const currentScope = scopeRef.current
    const liveProfile = $activeGatewayProfile.get() ?? ''
    const liveGateway = $gateway.get()

    setDetailsLoading(prev => ({ ...prev, [sessionKey]: true }))
    setDetailsError(prev => ({ ...prev, [sessionKey]: null }))

    try {
      const details = await fetchInboxRequestDetails(sessionKey, liveProfile)

      // Guard: scope changed, gateway swapped, or component unmounted (seq is stale)
      if (!mountedRef.current || seq !== loadSeqRef.current || scopeRef.current !== currentScope
        || $gateway.get() !== liveGateway || ($activeGatewayProfile.get() ?? '') !== liveProfile) {
        return
      }

      if (details) {
        setDetailsCache(prev => ({ ...prev, [sessionKey]: details }))
      } else {
        setDetailsError(prev => ({ ...prev, [sessionKey]: 'Failed to load details' }))
      }
    } catch {
      if (!mountedRef.current || seq !== loadSeqRef.current || scopeRef.current !== currentScope
        || $gateway.get() !== liveGateway || ($activeGatewayProfile.get() ?? '') !== liveProfile) {
        return
      }
      setDetailsError(prev => ({ ...prev, [sessionKey]: 'Failed to load details' }))
    } finally {
      if (mountedRef.current && seq === loadSeqRef.current && scopeRef.current === currentScope
        && $gateway.get() === liveGateway && ($activeGatewayProfile.get() ?? '') === liveProfile) {
        setDetailsLoading(prev => ({ ...prev, [sessionKey]: false }))
      }
    }
  }, [])

  // The expanded row's request state can change with NO user action: an approval times out,
  // a request is withdrawn, a record is dismissed elsewhere. Re-read the detail when the row's
  // request state moves WHILE IT IS EXPANDED, so the panel never keeps offering controls for a
  // request that is already gone (live bug: "Approve once" lingered past expiry). The initial
  // expand is the effect below; this watcher only fires on a transition.
  const expandedFingerprint = useMemo(() => {
    const item = items.find(candidate => candidate.session_key === expandedKey)

    if (!item) {
      return ''
    }

    return [
      item.pending_approval ? item.pending_approval.count : 0,
      item.pending_clarify ? item.pending_clarify.count : 0,
      item.expired_request_count ?? 0,
      item.lanes.join(',')
    ].join('|')
  }, [expandedKey, items])

  useEffect(() => {
    if (expandedKey) {
      void loadDetails(expandedKey)
    }
  }, [expandedKey, loadDetails])

  const previousRequestStateRef = useRef('')

  useEffect(() => {
    const state = expandedKey ? expandedFingerprint : ''

    if (state === previousRequestStateRef.current) {
      return
    }

    const wasExpanded = previousRequestStateRef.current !== ''

    previousRequestStateRef.current = state

    // Both states non-empty: the row stayed visible and only its REQUEST state moved. A row
    // that left the list (scope/profile switch, filter change) is not a request transition —
    // the scope guard already invalidated it and re-expanding fetches fresh.
    if (expandedKey && wasExpanded && state !== '') {
      void loadDetails(expandedKey)
    }
  }, [expandedKey, expandedFingerprint, loadDetails])

  const handleRowSelect = useCallback((item: InboxItem) => {
    setExpandedKey(prev => prev === item.session_key ? null : item.session_key)
  }, [])

  const handleNeedsAttentionClick = useCallback(() => {
    setNeedsAttentionOnly(true)
    setCategory('all')
  }, [])

  const handleCategoryClick = useCallback((cat: InboxCategory) => {
    setCategory(cat)
    setNeedsAttentionOnly(false)
  }, [])

  const openSession = useCallback((item: InboxItem) => {
    setSelectedStoredSessionId(item.session_key)
    navigate(sessionRoute(item.session_key))
    onClose()
  }, [navigate, onClose])

  const handleRetry = useCallback(() => {
    void refreshInbox(profile)
  }, [profile])

  const handleDetailRefresh = useCallback(() => {
    if (expandedKey) {
      setDetailsCache(prev => {
        const next = { ...prev }
        delete next[expandedKey]

        return next
      })
      void loadDetails(expandedKey)
    }
  }, [expandedKey, loadDetails])

  const coverageLine = coverage
    ? `Profile: ${coverage.profile} · ${coverage.scanned_sessions} sessions scanned`
    : 'loading connection…'

  const hasErrors = (coverage?.errors.length ?? 0) > 0
  const isPartial = coverage?.partial ?? false
  const isDisconnected = inbox.error !== null && inbox.capability !== 'unsupported'
  const isUnsupported = inbox.capability === 'unsupported'
  const isLoading = snapshot === null && !inbox.error && !isUnsupported

  // Compute nav counts from items
  const navCounts = useMemo(() => ({
    all: items.length,
    goals: countByCategory(items, 'goals'),
    loops: countByCategory(items, 'loops'),
    heartbeats: countByCategory(items, 'heartbeats'),
    background_tasks: countByCategory(items, 'background_tasks'),
    subagents: countByCategory(items, 'subagents'),
    other: countByCategory(items, 'other')
  }), [items])

  const navItems = useMemo(() => {
    const hasOther = navCounts.other > 0

    return CATEGORY_NAV.filter(cat => cat.id !== 'other' || hasOther)
  }, [navCounts.other])

  return (
    <Panel contentClassName={cn('flex h-full min-h-0 flex-col')} onClose={onClose}>
      <PanelHeader
        subtitle={coverageLine}
        title="Agent Inbox"
      />

      <PanelBody>
        {/* Left: compact navigation */}
        <div className="flex w-full shrink-0 flex-col gap-0.5 min-[47.5rem]:w-44">
          {/* Needs attention always first */}
          <button
            aria-expanded={needsAttentionOnly}
            className={cn(
              'flex h-7 items-center gap-2 rounded-md px-2 text-[0.78rem] transition-colors',
              needsAttentionOnly
                ? 'bg-accent/55 text-foreground font-medium'
                : 'text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-foreground'
            )}
            onClick={handleNeedsAttentionClick}
            type="button"
          >
            <span className={cn('size-1.5 shrink-0 rounded-full', needsAttentionCount > 0 ? 'bg-destructive' : 'bg-muted-foreground/30')} />
            <span className="flex-1 text-left">Needs attention</span>
            {needsAttentionCount > 0 && (
              <span className="tabular-nums text-[0.62rem] text-muted-foreground/60">{needsAttentionCount}</span>
            )}
          </button>

          <div className="my-1 h-px bg-(--ui-stroke-tertiary)" />

          {navItems.map(cat => {
            const count = navCounts[cat.id] ?? 0
            const isActive = category === cat.id && !needsAttentionOnly

            return (
              <button
                className={cn(
                  'flex h-7 items-center gap-2 rounded-md px-2 text-[0.78rem] transition-colors',
                  isActive
                    ? 'bg-accent/55 text-foreground font-medium'
                    : 'text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-foreground'
                )}
                key={cat.id}
                onClick={() => handleCategoryClick(cat.id)}
                type="button"
              >
                <Codicon className="shrink-0 text-muted-foreground/55" name={cat.icon} size="0.85rem" />
                <span className="flex-1 text-left">{cat.label}</span>
                {count > 0 && (
                  <span className="tabular-nums text-[0.62rem] text-muted-foreground/60">{count}</span>
                )}
              </button>
            )
          })}
        </div>

        {/* Right: session list with inline expansion */}
        <div className="flex min-h-0 flex-1 flex-col min-[47.5rem]:gap-4">
          {/* Search bar - full width */}
          <div className="mb-1 flex shrink-0 items-center gap-2">
            <SearchField
              aria-label="Search sessions"
              containerClassName="flex-1"
              onChange={setQuery}
              placeholder="Search title, key, or path…"
              value={query}
            />
            <SegmentedControl
              onChange={setSearchScope}
              options={[
                { id: 'section', label: 'This section' },
                { id: 'all', label: 'All sessions' }
              ]}
              value={searchScope}
            />
          </div>

          {/* Session list */}
          <div className="flex min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain">
            {isUnsupported ? (
              <PanelEmpty
                description="This connection or profile does not expose the inbox aggregation."
                icon="warning"
                title="Inbox not supported"
              />
            ) : isLoading ? (
              <PanelEmpty description="Reading the active profile…" icon="loading~spin" title="Loading inbox…" />
            ) : isDisconnected ? (
              <PanelEmpty
                action={
                  <Button disabled={inbox.loading} onClick={handleRetry} size="xs" variant="secondary">
                    <Codicon name="refresh" size="0.75rem" />
                    Retry
                  </Button>
                }
                description={inbox.error ?? 'Could not reach the gateway.'}
                icon="warning"
                title="Disconnected"
              />
            ) : hasErrors ? (
              <>
                <PanelEmpty
                  action={
                    <Button disabled={inbox.loading} onClick={handleRetry} size="xs" variant="secondary">
                      <Codicon name="refresh" size="0.75rem" />
                      Refresh
                    </Button>
                  }
                  description={`${coverage!.errors.length} session snapshot${coverage!.errors.length === 1 ? '' : 's'} could not be read.`}
                  icon="warning"
                  title="Partial read"
                />
                {visibleItems.map(item => (
                  <div className="flex flex-col" key={item.session_key}>
                    <PanelListRow
                      active={expandedKey === item.session_key}
                      dotClassName={laneDotClassName(item)}
                      expanded={expandedKey === item.session_key}
                      meta={metaLine(item)}
                      onSelect={() => handleRowSelect(item)}
                      rowKey={item.session_key}
                      title={item.title || item.session_key}
                    />
                    {expandedKey === item.session_key && (
                      <div className="ml-4 border-l-2 border-(--ui-stroke-tertiary) pl-3 pr-2 pb-2">
                        <InlineDetail
                          detailsError={detailsError[item.session_key]}
                          detailsLoading={detailsLoading[item.session_key]}
                          expandedDetails={detailsCache[item.session_key]}
                          item={item}
                          onOpenSession={openSession}
                          onRetry={handleDetailRefresh}
                        />
                      </div>
                    )}
                  </div>
                ))}
              </>
            ) : visibleItems.length === 0 && query.trim() ? (
              <PanelEmpty
                description={`No sessions matching "${query.trim()}"${searchScope === 'section' ? ` in ${needsAttentionOnly ? 'needs attention' : category === 'all' ? 'all sessions' : category}` : ''}.`}
                icon="search"
                title="No results"
              />
            ) : visibleItems.length === 0 ? (
              <PanelEmpty
                action={
                  isPartial ? (
                    <Button disabled={inbox.loading} onClick={handleRetry} size="xs" variant="secondary">
                      <Codicon name="refresh" size="0.75rem" />
                      Refresh
                    </Button>
                  ) : undefined
                }
                description={
                  isPartial
                    ? 'Some sessions could not be read. Counts may be incomplete.'
                    : `Nothing in ${needsAttentionOnly ? 'needs attention' : category === 'all' ? 'this profile' : category}.`
                }
                icon={isPartial ? 'warning' : 'inbox'}
                title={isPartial ? 'Incomplete data' : 'All clear'}
              />
            ) : (
              visibleItems.map(item => (
                <div className="flex flex-col" key={item.session_key}>
                  <PanelListRow
                    active={expandedKey === item.session_key}
                    dotClassName={laneDotClassName(item)}
                    expanded={expandedKey === item.session_key}
                    meta={metaLine(item)}
                    onSelect={() => handleRowSelect(item)}
                    rowKey={item.session_key}
                    title={item.title || item.session_key}
                  />
                  {expandedKey === item.session_key && (
                    <div className="ml-4 border-l-2 border-(--ui-stroke-tertiary) pl-3 pr-2 pb-2">
                      <InlineDetail
                        detailsError={detailsError[item.session_key]}
                        detailsLoading={detailsLoading[item.session_key]}
                        expandedDetails={detailsCache[item.session_key]}
                        item={item}
                        onOpenSession={openSession}
                        onRetry={handleDetailRefresh}
                      />
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </PanelBody>
    </Panel>
  )
}
