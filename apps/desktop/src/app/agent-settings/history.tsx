import './history.css'

import { type InfiniteData, useInfiniteQuery, useQuery, useQueryClient } from '@tanstack/react-query'
import { Fragment, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import { $apiRequestScope } from '@/api/client'
import { ActionsMenu, renderActionItem } from '@/components/ui/actions-menu'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Codicon } from '@/components/ui/codicon'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { SearchField } from '@/components/ui/search-field'
import { useI18n } from '@/i18n'
import { pathLabel, sessionTitle } from '@/lib/chat-runtime'
import { sessionSourceLabel } from '@/lib/session-source'
import { $activeConnectionId } from '@/store/connections'
import { $activeGatewayProfile } from '@/store/profile'
import type { PaginatedSessions, SessionInfo } from '@/types/hermes'

import { openSession } from '../open-session'

import { agentSettingsKey, type AgentSettingsOwner } from './data'
import { useHistoryCopy } from './history-copy'
import {
  DEFAULT_HISTORY_FILTERS,
  getHistoryPage,
  getHistoryRunning,
  type HistoryDeleteOutcome,
  type HistoryFilters,
  historyRows,
  historyRunning,
  historySource,
  matchesHistory,
  mutateHistorySession,
  nextHistoryOffset
} from './history-data'
import {
  HistoryCleanupDialog,
  HistoryDeleteDialog,
  HistoryError,
  HistoryPreview,
  HistoryRename
} from './history-dialogs'

export interface AgentHistoryProps {
  owner: AgentSettingsOwner
  onClose: () => void
}

/** Redesign-only caller supplies the exact owner; remounting discards selections and confirmations. */
export function AgentHistory(props: AgentHistoryProps) {
  const c = useHistoryCopy()

  if (!props.owner.connectionId) {
    return <HistoryError error={c.unknown} />
  }

  return <AgentHistoryContent key={JSON.stringify(agentSettingsKey(props.owner))} {...props} />
}

function AgentHistoryContent({ owner, onClose }: AgentHistoryProps) {
  const c = useHistoryCopy()
  const { t } = useI18n()
  const navigate = useNavigate()
  const client = useQueryClient()
  const mounted = useRef(true)
  const [filters, setFilters] = useState<HistoryFilters>(DEFAULT_HISTORY_FILTERS)
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [preview, setPreview] = useState<SessionInfo | null>(null)
  const [rename, setRename] = useState<SessionInfo | null>(null)
  const [deleting, setDeleting] = useState<SessionInfo[] | null>(null)
  const [cleanup, setCleanup] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const [outcome, setOutcome] = useState<HistoryDeleteOutcome | null>(null)
  useLayoutEffect(() => {
    mounted.current = true

    return () => {
      mounted.current = false
    }
  }, [])

  const current = useCallback(
    () =>
      mounted.current &&
      ($apiRequestScope.get().connectionId ?? $activeConnectionId.get() ?? 'local') === owner.connectionId &&
      $activeGatewayProfile.get() === owner.profile,
    [owner.connectionId, owner.profile]
  )

  const historyKey = [...agentSettingsKey(owner), 'history']

  const recent = useInfiniteQuery({
    queryKey: [...historyKey, filters.source, filters.archived],
    initialPageParam: 0,
    queryFn: ({ pageParam }) => getHistoryPage(owner, filters, pageParam),
    getNextPageParam: nextHistoryOffset
  })

  const running = useQuery({
    queryKey: [...agentSettingsKey(owner), 'history-running'],
    queryFn: () => getHistoryRunning(owner),
    refetchInterval: 10_000
  })

  const rows = useMemo(
    () =>
      historyRows(recent.data?.pages ?? [])
        .filter(row => matchesHistory(row, filters, search))
        .sort((a, b) => Number(Boolean(b.pinned)) - Number(Boolean(a.pinned)) || b.last_active - a.last_active),
    [recent.data, filters, search]
  )

  const selectedRows = rows.filter(row => selected.has(row.id))
  const { hasNextPage, isFetching, isError, fetchNextPage } = recent
  // Search is eventually exhaustive, not a capped FTS response masquerading as all history.
  // Source/archive filters run on the server; title/preview matching walks bounded pages.
  useEffect(() => {
    if (search.trim() && hasNextPage && !isFetching && !isError && !busy) {
      void fetchNextPage()
    }
  }, [search, hasNextPage, isFetching, isError, fetchNextPage, busy, recent.data?.pages.length])

  function changeFilters(next: HistoryFilters) {
    setFilters(next)
    setSelected(new Set())
  }

  function open(row: SessionInfo) {
    if (!current()) {
      return
    }

    onClose()
    openSession(row.id, navigate, 'in-place', {
      workspaceMode: 'sessions',
      ownerRoute: { connectionId: owner.connectionId!, profile: owner.profile }
    })
  }

  async function mutate(selection: SessionInfo | SessionInfo[], patch: Partial<SessionInfo>) {
    if (busy || !current()) {
      return
    }

    setBusy(true)
    setError(null)
    await client.cancelQueries({ queryKey: historyKey })

    if (!current()) {
      return
    }

    const targets = Array.isArray(selection) ? selection : [selection]
    const ids = new Set(targets.map(row => row.id))
    const saved = new Set<string>()
    const before = client.getQueriesData<InfiniteData<PaginatedSessions>>({ queryKey: historyKey })
    client.setQueriesData<InfiniteData<PaginatedSessions>>(
      { queryKey: historyKey },
      data =>
        data && {
          ...data,
          pages: data.pages.map(page => ({
            ...page,
            sessions: page.sessions.map(item => (ids.has(item.id) ? { ...item, ...patch } : item))
          }))
        }
    )

    try {
      for (const row of targets) {
        const result = await mutateHistorySession(owner, row.id, 'PATCH', patch, current, c.upgrade)

        if (!result.ok) {
          throw new Error(c.errorHint)
        }

        saved.add(row.id)
      }
    } catch (err) {
      for (const [key, data] of before) {
        client.setQueryData(
          key,
          data && {
            ...data,
            pages: data.pages.map(page => ({
              ...page,
              sessions: page.sessions.map(item => (saved.has(item.id) ? { ...item, ...patch } : item))
            }))
          }
        )
      }

      throw err
    } finally {
      if (current()) {
        setBusy(false)
        void client.invalidateQueries({ queryKey: historyKey })
        void client.invalidateQueries({ queryKey: [...agentSettingsKey(owner), 'recent'] })
      }
    }
  }

  function deleted(result: HistoryDeleteOutcome) {
    if (!current()) {
      return
    }

    setOutcome(result)
    setSelected(new Set())
    const removed = new Set(result.deleted)
    client.setQueriesData<InfiniteData<PaginatedSessions>>(
      { queryKey: historyKey },
      data =>
        data && {
          ...data,
          pages: data.pages.map(page => ({ ...page, sessions: page.sessions.filter(row => !removed.has(row.id)) }))
        }
    )
    void client.invalidateQueries({ queryKey: historyKey })
    void client.invalidateQueries({ queryKey: [...agentSettingsKey(owner), 'recent'] })
  }

  return (
    <section aria-label={c.title} className="agent-history">
      <header className="agent-history-heading">
        <h2 className="sr-only">{c.title}</h2>
        <p>
          {c.sessions}: {rows.length} · {c.messages}: {rows.reduce((sum, row) => sum + row.message_count, 0)} ·{' '}
          {c.archived}: {rows.filter(row => row.archived).length}
        </p>
        <p className="agent-history-help">{c.description}</p>
      </header>
      <div className="agent-history-toolbar">
        <SearchField
          containerClassName="agent-history-search"
          loading={Boolean(search && recent.isFetching)}
          onChange={value => {
            setSearch(value)
            setSelected(new Set())
          }}
          placeholder={c.search}
          value={search}
          variant="box"
        />
        <HistoryFilterPopover filters={filters} onChange={changeFilters} />
        <Button disabled={busy} onClick={() => setCleanup(true)} variant="secondary">
          <Codicon name="trash" />
          {c.cleanup}
        </Button>
      </div>
      <HistoryFilterChips filters={filters} onChange={changeFilters} />
      {recent.isError && <HistoryError error={recent.error} retry={() => void recent.refetch()} />}
      {running.isError && <HistoryError error={running.error} retry={() => void running.refetch()} />}
      {Boolean(error) && (
        <HistoryError
          error={error}
          retry={() => {
            setError(null)
            void recent.refetch()
          }}
        />
      )}
      {outcome && <HistoryOutcome outcome={outcome} />}
      {selectedRows.length > 0 && (
        <div className="agent-history-selection">
          <span>
            {c.selected}: {selectedRows.length}
          </span>
          <Button
            disabled={busy}
            onClick={() => void mutate(selectedRows, { pinned: true }).catch(setError)}
            size="sm"
            variant="secondary"
          >
            <Codicon name="pin" />
            {c.pin}
          </Button>
          <Button
            disabled={busy}
            onClick={() => void mutate(selectedRows, { archived: true, pinned: false }).catch(setError)}
            size="sm"
            variant="secondary"
          >
            <Codicon name="archive" />
            {c.archive}
          </Button>
          <Button disabled={busy} onClick={() => setDeleting(selectedRows)} size="sm" variant="destructive">
            {c.delete}
          </Button>
          <Button disabled={busy} onClick={() => setSelected(new Set())} size="sm" variant="ghost">
            {t.common.clear}
          </Button>
        </div>
      )}
      <div className="agent-history-body">
        <div className="agent-history-list">
          {!rows.length && !recent.isPending && !recent.isError && (
            <p className="agent-history-help">{recent.hasNextPage ? c.partial : c.empty}</p>
          )}
          <HistoryRows
            busy={busy}
            filters={filters}
            onDelete={row => setDeleting([row])}
            onOpen={open}
            onPreview={setPreview}
            onRename={setRename}
            onToggle={row =>
              setSelected(previous => {
                const next = new Set(previous)

                if (next.has(row.id)) {
                  next.delete(row.id)
                } else {
                  next.add(row.id)
                }

                return next
              })
            }
            onToggleAll={checked => setSelected(checked ? new Set(rows.map(row => row.id)) : new Set())}
            onUpdate={(row, patch) => {
              void mutate(row, patch).catch(setError)
            }}
            rows={rows}
            running={running.data ?? new Set()}
            selected={selected}
          />
          <HistoryScope
            hasNextPage={recent.hasNextPage}
            isError={recent.isError}
            isPending={recent.isPending}
            search={search}
          />
          {recent.hasNextPage && (
            <div className="agent-history-footer">
              <Button
                disabled={recent.isFetching || busy}
                onClick={() => void recent.fetchNextPage()}
                variant="secondary"
              >
                {recent.isFetching ? c.loading : c.more}
              </Button>
            </div>
          )}
        </div>
        {preview && (
          <HistoryPreview
            key={preview.id}
            onArchive={() =>
              mutate(preview, { archived: !preview.archived, ...(!preview.archived ? { pinned: false } : {}) })
            }
            onClose={() => setPreview(null)}
            onDelete={() => {
              setDeleting([preview])
              setPreview(null)
            }}
            onOpen={() => open(preview)}
            owner={owner}
            row={preview}
          />
        )}
      </div>
      {rename && (
        <HistoryRename onClose={() => setRename(null)} onSave={title => mutate(rename, { title })} row={rename} />
      )}
      {deleting && (
        <HistoryDeleteDialog
          current={current}
          onClose={() => setDeleting(null)}
          onDeleted={deleted}
          owner={owner}
          rows={deleting}
        />
      )}
      {cleanup && (
        <HistoryCleanupDialog current={current} onClose={() => setCleanup(false)} onDeleted={deleted} owner={owner} />
      )}
    </section>
  )
}

interface HistoryScopeProps {
  hasNextPage: boolean
  isPending: boolean
  isError: boolean
  search: string
}

function HistoryScope({ hasNextPage, isPending, isError, search }: HistoryScopeProps) {
  const c = useHistoryCopy()

  return (
    <div className="agent-history-scope" role="status">
      {hasNextPage
        ? search.trim()
          ? c.searching
          : c.partial
        : isPending
          ? c.loading
          : isError
            ? c.unknown
            : c.complete}
    </div>
  )
}

function HistoryFilterChips({
  filters,
  onChange
}: {
  filters: HistoryFilters
  onChange: (filters: HistoryFilters) => void
}) {
  const c = useHistoryCopy()

  if (!filters.source.length && filters.archived === 'exclude' && filters.group === 'project') {
    return null
  }

  return (
    <div className="agent-history-filter-chips">
      {filters.source.map(source => (
        <Button
          key={source}
          onClick={() => onChange({ ...filters, source: filters.source.filter(value => value !== source) })}
          variant="secondary"
        >
          {c.source}: {{ chat: c.chat, schedule: c.schedule, telegram: 'Telegram', discord: 'Discord' }[source]}
          <Codicon name="close" />
        </Button>
      ))}
      {filters.archived !== 'exclude' && (
        <Button onClick={() => onChange({ ...filters, archived: 'exclude' })} variant="secondary">
          {c.archived}: {filters.archived === 'include' ? c.include : c.only}
          <Codicon name="close" />
        </Button>
      )}
      {filters.group !== 'project' && (
        <Button onClick={() => onChange({ ...filters, group: 'project' })} variant="secondary">
          {c.group}: {filters.group === 'source' ? c.source : c.none}
          <Codicon name="close" />
        </Button>
      )}
    </div>
  )
}

function HistoryFilterPopover({
  filters,
  onChange
}: {
  filters: HistoryFilters
  onChange: (filters: HistoryFilters) => void
}) {
  const c = useHistoryCopy()
  const [open, setOpen] = useState(false)

  const active =
    Number(filters.source.length > 0) + Number(filters.archived !== 'exclude') + Number(filters.group !== 'project')

  return (
    <Popover onOpenChange={setOpen} open={open}>
      <PopoverTrigger asChild>
        <Button className="agent-history-filter-trigger" data-active={active > 0 || undefined} variant="secondary">
          <Codicon name="filter" />
          {c.filters}
          {active > 0 && <span className="agent-history-filter-count text-primary-foreground">{active}</span>}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="agent-history-filter-popover" variant="menu">
        <fieldset className="agent-history-filter-options">
          <legend>{c.source}</legend>
          {(['chat', 'schedule', 'telegram', 'discord'] as const).map(source => (
            <label className="agent-history-checkbox" key={source}>
              <Checkbox
                checked={filters.source.includes(source)}
                onCheckedChange={checked =>
                  onChange({
                    ...filters,
                    source: checked ? [...filters.source, source] : filters.source.filter(value => value !== source)
                  })
                }
              />
              {{ chat: c.chat, schedule: c.schedule, telegram: 'Telegram', discord: 'Discord' }[source]}
            </label>
          ))}
        </fieldset>
        <fieldset className="agent-history-filter-options">
          <legend>{c.archived}</legend>
          {(['exclude', 'include', 'only'] as const).map(archived => (
            <label className="agent-history-checkbox" key={archived}>
              <input
                checked={filters.archived === archived}
                name="history-archived"
                onChange={() => onChange({ ...filters, archived })}
                type="radio"
              />
              {{ exclude: c.hide, include: c.include, only: c.only }[archived]}
            </label>
          ))}
        </fieldset>
        <fieldset className="agent-history-filter-options">
          <legend>{c.group}</legend>
          {(['project', 'source', 'none'] as const).map(group => (
            <label className="agent-history-checkbox" key={group}>
              <input
                checked={filters.group === group}
                name="history-group"
                onChange={() => onChange({ ...filters, group })}
                type="radio"
              />
              {{ project: c.project, source: c.source, none: c.none }[group]}
            </label>
          ))}
        </fieldset>
        <footer>
          <Button onClick={() => onChange(DEFAULT_HISTORY_FILTERS)} variant="ghost">
            {c.reset}
          </Button>
          <Button onClick={() => setOpen(false)} variant="outline">
            {c.done}
          </Button>
        </footer>
      </PopoverContent>
    </Popover>
  )
}

interface HistoryRowsProps {
  rows: SessionInfo[]
  filters: HistoryFilters
  selected: Set<string>
  running: Set<string>
  busy: boolean
  onToggle: (row: SessionInfo) => void
  onOpen: (row: SessionInfo) => void
  onPreview: (row: SessionInfo) => void
  onToggleAll: (checked: boolean) => void
  onRename: (row: SessionInfo) => void
  onDelete: (row: SessionInfo) => void
  onUpdate: (row: SessionInfo, patch: Partial<SessionInfo>) => void
}

function HistoryRows(props: HistoryRowsProps) {
  const c = useHistoryCopy()
  const { locale } = useI18n()
  const labels: Record<string, string> = { chat: c.chat, schedule: c.schedule }
  const sourceName = (row: SessionInfo) => labels[historySource(row)] ?? sessionSourceLabel(row.source) ?? c.other
  const groups = new Map<string, SessionInfo[]>()

  for (const row of props.rows) {
    const key =
      props.filters.group === 'source'
        ? sourceName(row)
        : props.filters.group === 'project'
          ? row.git_repo_root || row.cwd || c.noProject
          : ''

    const group = groups.get(key)

    if (group) {
      group.push(row)
    } else {
      groups.set(key, [row])
    }
  }

  return (
    <div className="agent-history-table-scroll">
      <table className="agent-history-table">
        <thead>
          <tr>
            <th>
              <Checkbox
                aria-label={c.selectAll}
                checked={
                  props.rows.length > 0 && props.rows.every(row => props.selected.has(row.id))
                    ? true
                    : props.rows.some(row => props.selected.has(row.id))
                      ? 'indeterminate'
                      : false
                }
                disabled={!props.rows.length || props.busy}
                onCheckedChange={value => props.onToggleAll(value === true)}
              />
            </th>
            <th>{c.sessionTitle}</th>
            <th>{c.source}</th>
            <th className="agent-history-project">{c.project}</th>
            <th>{c.messages}</th>
            <th className="agent-history-cost">{c.cost}</th>
            <th>{c.latest}</th>
            <th>
              <span className="sr-only">{c.actions}</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {[...groups].map(([group, rows]) => (
            <Fragment key={group}>
              {group && (
                <tr className="agent-history-group">
                  <th colSpan={8} scope="rowgroup" title={group}>
                    <Codicon name={props.filters.group === 'project' ? 'folder' : 'comment-discussion'} />
                    {props.filters.group === 'project' ? pathLabel(group) : group} <span>{rows.length}</span>
                  </th>
                </tr>
              )}
              {rows.map(row => (
                <tr className="agent-history-row" data-selected={props.selected.has(row.id)} key={row.id}>
                  <td>
                    <Checkbox
                      aria-label={`${c.selected}: ${sessionTitle(row)}`}
                      checked={props.selected.has(row.id)}
                      disabled={props.busy}
                      onCheckedChange={() => props.onToggle(row)}
                    />
                  </td>
                  <td>
                    <button className="agent-history-row-main" onClick={() => props.onOpen(row)} type="button">
                      <span className="agent-history-row-title">
                        {row.pinned && <Codicon name="pin" />}
                        {sessionTitle(row)}
                      </span>
                      {row.message_count === 0 && <small>{c.emptyMessages}</small>}
                      {row.archived && <small>{c.archived}</small>}
                      {historyRunning(row, props.running) && <small>{c.running}</small>}
                    </button>
                  </td>
                  <td>{sourceName(row)}</td>
                  <td className="agent-history-project" title={row.git_repo_root || row.cwd || ''}>
                    {row.git_branch || (row.cwd ? pathLabel(row.cwd) : '—')}
                  </td>
                  <td>{row.message_count}</td>
                  <td className="agent-history-cost">
                    {row.actual_cost_usd == null && row.estimated_cost_usd == null
                      ? '—'
                      : new Intl.NumberFormat(locale, { style: 'currency', currency: 'USD' }).format(
                          row.actual_cost_usd ?? row.estimated_cost_usd!
                        )}
                  </td>
                  <td>
                    {row.last_active > 0
                      ? new Date(row.last_active * 1000).toLocaleDateString(locale, { month: 'short', day: 'numeric' })
                      : '—'}
                  </td>
                  <td>
                    <ActionsMenu
                      items={kit => (
                        <>
                          {[
                            { label: c.open, icon: 'comment-discussion', onSelect: () => props.onOpen(row) },
                            { label: c.preview, icon: 'preview', onSelect: () => props.onPreview(row) },
                            {
                              label: c.rename,
                              icon: 'edit',
                              disabled: props.busy,
                              onSelect: () => props.onRename(row)
                            },
                            {
                              label: row.pinned ? c.unpin : c.pin,
                              icon: 'pin',
                              disabled: props.busy,
                              onSelect: () => props.onUpdate(row, { pinned: !row.pinned })
                            },
                            {
                              label: row.archived ? c.unarchive : c.archive,
                              icon: 'archive',
                              disabled: props.busy,
                              onSelect: () =>
                                props.onUpdate(row, {
                                  archived: !row.archived,
                                  ...(!row.archived ? { pinned: false } : {})
                                })
                            },
                            {
                              label: c.delete,
                              icon: 'trash',
                              disabled: props.busy,
                              variant: 'destructive' as const,
                              onSelect: () => props.onDelete(row)
                            }
                          ].map(action => renderActionItem(kit, action))}
                        </>
                      )}
                    >
                      <Button aria-label={`${c.actions}: ${sessionTitle(row)}`} size="icon-xs" variant="ghost">
                        <Codicon name="ellipsis" />
                      </Button>
                    </ActionsMenu>
                  </td>
                </tr>
              ))}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function HistoryOutcome({ outcome }: { outcome: HistoryDeleteOutcome }) {
  const c = useHistoryCopy()

  return (
    <div className="agent-history-outcome" role="status">
      <p>
        {c.deleted}: {outcome.deleted.length} · {c.running}: {outcome.running.length} · {c.changed}:{' '}
        {outcome.changed.length} · {c.errors}: {outcome.errors.length}
      </p>
      {outcome.errors.length > 0 && (
        <>
          <p>{c.errorHint}</p>
          <ul>
            {outcome.errors.map(error => (
              <li key={error.id}>
                {error.id}: {error.error}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  )
}
