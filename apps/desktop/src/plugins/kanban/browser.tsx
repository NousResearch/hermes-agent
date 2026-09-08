import { Badge, Button, Codicon, EmptyState, ErrorState, host, Loader, useQuery } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { type BoardBrowserReads, type BoardSource, browserFailure, browserKeys, browserRefresh } from './browser-api'
import { CardFace } from './card-face'
import { DescriptionSection, Diagnostics } from './drawer'
import { columnMeta } from './types'
import { ago, Avatar, columnLabel, duration, Section, shortId, useKanban } from './ui'

interface Selection {
  readonly connectionId: string
  readonly profile: string
  readonly board: string
}

function usePageVisible() {
  const [visible, setVisible] = useState(document.visibilityState !== 'hidden')

  useEffect(() => {
    const update = () => setVisible(document.visibilityState !== 'hidden')
    document.addEventListener('visibilitychange', update)

    return () => document.removeEventListener('visibilitychange', update)
  }, [])

  return visible
}

interface ReadState {
  dataUpdatedAt: number
  error: unknown
  isFetching: boolean
  refetch: () => unknown
}

function Freshness({ query, stale = false }: { query: ReadState; stale?: boolean }) {
  const k = useKanban().browser

  return (
    <div className="flex flex-col gap-1 text-xs text-(--ui-text-tertiary)" role="status">
      {!!query.error && <ErrorState title={k[browserFailure(query.error)]} />}
      {(stale || !!query.error) && query.dataUpdatedAt > 0 && <p className="text-(--ui-text-secondary)">{k.stale}</p>}
      <div className="flex flex-wrap items-center gap-2">
        <span>{query.dataUpdatedAt ? k.refreshed(new Date(query.dataUpdatedAt).toLocaleString()) : k.never}</span>
        <Button disabled={query.isFetching} onClick={() => void query.refetch()} size="xs" variant="ghost">
          <Codicon name="refresh" spinning={query.isFetching} />
          {query.error ? k.retry : k.refresh}
        </Button>
      </div>
    </div>
  )
}

const sourceLabel = (source: BoardSource) => `${source.label} / ${source.profile}`

const matches = (selection: Selection | null, source: BoardSource) =>
  selection?.connectionId === source.connectionId && selection.profile === source.profile

function SourceBoards({
  enabled,
  onSelect,
  reads,
  selection,
  source
}: {
  enabled: boolean
  onSelect: (selection: Selection) => void
  reads: BoardBrowserReads
  selection: Selection | null
  source: BoardSource
}) {
  const k = useKanban().browser

  const query = useQuery({
    ...browserRefresh,
    enabled,
    queryKey: browserKeys.boards(source),
    queryFn: ({ signal }) => reads.boards(source, signal)
  })

  return (
    <section aria-label={sourceLabel(source)} className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="break-words text-sm font-medium">{sourceLabel(source)}</h2>
        {query.isSuccess && (
          <Badge size="xs" variant="muted">
            {k.available}
          </Badge>
        )}
      </div>
      <Freshness query={query} stale={!enabled} />
      {query.isPending && !query.error && <Loader />}
      {query.data?.boards.length === 0 && <p className="text-xs text-(--ui-text-tertiary)">{k.emptyBoards}</p>}
      {query.data?.boards.map(board => (
        <Button
          aria-label={`${sourceLabel(source)} / ${board.name || board.slug}`}
          aria-pressed={matches(selection, source) && selection?.board === board.slug}
          className="justify-start text-left"
          key={board.slug}
          onClick={() =>
            onSelect(Object.freeze({ connectionId: source.connectionId, profile: source.profile, board: board.slug }))
          }
          size="sm"
          variant={matches(selection, source) && selection?.board === board.slug ? 'secondary' : 'ghost'}
        >
          <Codicon name="project" />
          <span className="truncate">{board.name || board.slug}</span>
          {board.total != null && <span className="ml-auto tabular-nums text-(--ui-text-tertiary)">{board.total}</span>}
        </Button>
      ))}
    </section>
  )
}

function BrowserTask({
  board,
  enabled,
  id,
  onClose,
  onOpen,
  reads,
  source
}: {
  board: string
  enabled: boolean
  id: string
  onClose: () => void
  onOpen: (id: string) => void
  reads: BoardBrowserReads
  source: BoardSource
}) {
  const k = useKanban()

  const query = useQuery({
    ...browserRefresh,
    enabled,
    queryKey: browserKeys.task(source, board, id),
    queryFn: ({ signal }) => reads.task(source, board, id, signal)
  })

  const detail = query.data

  return (
    <aside
      aria-label={`${sourceLabel(source)} / ${board} / ${id}`}
      className="flex w-96 max-w-full shrink-0 flex-col gap-4 overflow-y-auto border-l border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) p-4"
      onKeyDown={event => {
        if (event.key === 'Escape') {
          event.stopPropagation()
          onClose()
        }
      }}
    >
      <header className="flex items-center gap-2">
        <Badge variant="muted">{k.browser.readOnly}</Badge>
        <span className="min-w-0 break-words text-xs text-(--ui-text-tertiary)">
          {sourceLabel(source)} / {board}
        </span>
        <Button aria-label={k.close} className="ml-auto" onClick={onClose} size="icon-xs" variant="ghost">
          <Codicon name="close" />
        </Button>
      </header>
      <Freshness query={query} stale={!enabled} />
      {query.isPending && !query.error && <Loader />}
      {detail && (
        <>
          <h2 className="text-sm font-semibold">{detail.task.title || id}</h2>
          <div className="flex flex-wrap items-center gap-2 text-xs text-(--ui-text-secondary)">
            <Badge variant="muted">{columnLabel(k, detail.task.status)}</Badge>
            {detail.task.assignee && <Avatar name={detail.task.assignee} />}
            <span>{detail.task.assignee || k.unassigned}</span>
            <span className="font-mono">{shortId(id)}</span>
          </div>
          {!!detail.task.diagnostics?.length && <Diagnostics items={detail.task.diagnostics} />}
          <DescriptionSection body={detail.task.body} />
          {detail.task.result && (
            <Section label={k.result}>
              <p className="whitespace-pre-wrap text-sm">{detail.task.result}</p>
            </Section>
          )}
          {detail.task.latest_summary && (
            <Section label={k.latestSummary}>
              <p className="whitespace-pre-wrap text-sm">{detail.task.latest_summary}</p>
            </Section>
          )}
          <Section label={k.dependencies}>
            {(['parents', 'children'] as const).map(
              side =>
                detail.links[side].length > 0 && (
                  <div className="flex flex-wrap items-center gap-2 text-xs" key={side}>
                    <span>{side === 'parents' ? k.blockedBy : k.blocks}</span>
                    {detail.links[side].map(linked => (
                      <Button key={linked} onClick={() => onOpen(linked)} size="xs" variant="ghost">
                        {shortId(linked)}
                      </Button>
                    ))}
                  </div>
                )
            )}
          </Section>
          <Section label={k.runs(detail.runs.length)}>
            {detail.runs.map(run => (
              <div className="flex flex-col gap-1 text-xs" key={run.id}>
                <div className="flex flex-wrap items-center gap-2">
                  <Badge size="xs" variant={run.error ? 'destructive' : 'muted'}>
                    {run.outcome || run.status}
                  </Badge>
                  <span>{run.profile}</span>
                  <span>{duration(run.started_at, run.ended_at)}</span>
                  <span>{ago(run.ended_at || run.started_at)}</span>
                </div>
                <p className="whitespace-pre-wrap text-(--ui-text-secondary)">{run.error || run.summary}</p>
              </div>
            ))}
          </Section>
        </>
      )}
    </aside>
  )
}

function SelectedBoard({
  enabled,
  reads,
  selection,
  source
}: {
  enabled: boolean
  reads: BoardBrowserReads
  selection: Selection
  source: BoardSource
}) {
  const k = useKanban()
  const [task, setTask] = useState<string | null>(null)

  const inventory = useQuery({
    ...browserRefresh,
    enabled,
    queryKey: browserKeys.boards(source),
    queryFn: ({ signal }) => reads.boards(source, signal)
  })

  const board = inventory.data?.boards.find(board => board.slug === selection.board)

  const query = useQuery({
    ...browserRefresh,
    enabled: enabled && !!board && !inventory.error,
    queryKey: browserKeys.board(source, selection.board),
    queryFn: ({ signal }) => reads.board(source, selection.board, signal)
  })

  if (inventory.isSuccess && !board) {
    return <EmptyState title={k.browser.removed} />
  }

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <header className="flex flex-col gap-2 p-4">
        <h2 className="text-sm font-semibold">
          {sourceLabel(source)} / {board?.name || selection.board}
        </h2>
        <Freshness query={query} stale={!enabled || !!inventory.error} />
      </header>
      <div className="flex min-h-0 flex-1 overflow-hidden">
        <div className="flex min-w-0 flex-1 gap-3 overflow-auto p-4 pt-0">
          {query.isPending && !query.error && <Loader />}
          {query.data?.columns.map(column => (
            <section
              aria-label={columnLabel(k, column.name)}
              className="flex w-64 shrink-0 flex-col gap-2"
              key={column.name}
            >
              <h3 className="flex items-center gap-2 text-xs font-medium uppercase text-(--ui-text-tertiary)">
                <span className="size-1.5 rounded-full" style={{ backgroundColor: columnMeta(column.name).tone }} />
                {columnLabel(k, column.name)}
                <span>{column.tasks.length}</span>
              </h3>
              {column.tasks.map(card => (
                <CardFace
                  aria-label={card.title || card.id}
                  className="cursor-pointer"
                  key={card.id}
                  onClick={() => setTask(card.id)}
                  onKeyDown={event => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault()
                      setTask(card.id)
                    }
                  }}
                  role="button"
                  tabIndex={0}
                  task={card}
                >
                  <div className="flex items-center gap-2 text-xs text-(--ui-text-tertiary)">
                    {card.assignee && <Avatar name={card.assignee} size="1.125rem" />}
                    <span>{card.assignee || k.unassigned}</span>
                    {!!card.warnings?.count && (
                      <Badge size="xs" variant="warn">
                        {card.warnings.count}
                      </Badge>
                    )}
                    <span className="ml-auto font-mono">{shortId(card.id)}</span>
                  </div>
                </CardFace>
              ))}
              {!column.tasks.length && <p className="text-xs text-(--ui-text-quaternary)">{k.empty}</p>}
            </section>
          ))}
          {query.data && query.data.columns.every(column => column.tasks.length === 0) && (
            <EmptyState title={k.noTasks} />
          )}
        </div>
        {task && (
          <BrowserTask
            board={selection.board}
            enabled={enabled && !!board && !inventory.error && !query.error}
            id={task}
            key={task}
            onClose={() => setTask(null)}
            onOpen={setTask}
            reads={reads}
            source={source}
          />
        )}
      </div>
    </div>
  )
}

/** A separate read-only route in the bundled plugin. Never mounts the editable
 *  board controller, its socket, orchestration hooks, or mutation helpers. */
export function BoardBrowser({ reads }: { reads: BoardBrowserReads }) {
  const k = useKanban().browser
  const visible = usePageVisible()
  const [selection, setSelection] = useState<Selection | null>(null)

  const inventory = useQuery({
    ...browserRefresh,
    enabled: visible,
    queryKey: browserKeys.inventory,
    queryFn: ({ signal }) => reads.sources(signal)
  })

  const source = inventory.data?.find(source => matches(selection, source))
  const enabled = visible && !inventory.error

  return (
    <div className="flex h-full min-h-0 flex-col bg-(--ui-surface-background)">
      <header className="flex flex-wrap items-center gap-2 px-4 py-3">
        <h1 className="text-sm font-semibold">{k.title}</h1>
        <Badge variant="muted">{k.readOnly}</Badge>
        <Button className="ml-auto" onClick={() => host.navigate('/kanban')} size="sm" variant="ghost">
          {k.back}
        </Button>
      </header>
      <div className="flex min-h-0 flex-1">
        <nav
          aria-label={k.sources}
          className="flex w-72 shrink-0 flex-col gap-5 overflow-y-auto border-r border-(--ui-stroke-tertiary) p-4"
        >
          <Freshness query={inventory} />
          <p className="text-xs text-(--ui-text-tertiary)">{k.ownership}</p>
          {inventory.isPending && !inventory.error && <Loader />}
          {inventory.data?.length === 0 && <EmptyState title={k.emptySources} />}
          {inventory.data?.map(source => (
            <SourceBoards
              enabled={enabled}
              key={JSON.stringify([source.connectionId, source.profile])}
              onSelect={setSelection}
              reads={reads}
              selection={selection}
              source={source}
            />
          ))}
        </nav>
        {selection && source ? (
          <SelectedBoard
            enabled={enabled}
            key={JSON.stringify(selection)}
            reads={reads}
            selection={selection}
            source={source}
          />
        ) : (
          <EmptyState title={selection ? k.removed : k.choose} />
        )}
      </div>
    </div>
  )
}
