import type * as React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { archiveWorkQueueItem, createWorkQueueItem, getWorkQueue, snoozeWorkQueueItem, updateWorkQueueItem, type WorkQueueItem } from '@/hermes'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { OverlayView } from '../overlays/overlay-view'

interface WorkQueueViewProps {
  onClose: () => void
}

const STATUS_LABELS: Record<string, string> = {
  blocked: 'Blocked',
  done: 'Done',
  failed: 'Failed',
  needs_review: 'Needs Vlad',
  running: 'Running',
  scheduled: 'Scheduled',
  snoozed: 'Snoozed'
}

const STATUS_TONE: Record<string, string> = {
  blocked: 'bg-amber-500/10 text-amber-600 dark:text-amber-300',
  done: 'bg-muted text-muted-foreground',
  failed: 'bg-destructive/10 text-destructive',
  needs_review: 'bg-primary/10 text-primary',
  running: 'bg-sky-500/10 text-sky-600 dark:text-sky-300',
  scheduled: 'bg-muted text-muted-foreground',
  snoozed: 'bg-muted text-muted-foreground'
}

const SECTIONS = [
  { id: 'needs', label: 'Needs Vlad', statuses: ['needs_review', 'blocked', 'failed'] },
  { id: 'running', label: 'Running', statuses: ['running'] },
  { id: 'scheduled', label: 'Scheduled', statuses: ['scheduled', 'snoozed'] },
  { id: 'done', label: 'Done', statuses: ['done'] }
] as const

function formatWhen(value?: null | string): string {
  if (!value) {
    return '—'
  }

  const date = new Date(value)

  if (Number.isNaN(date.getTime())) {
    return value
  }

  return date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

function statusLabel(status: string): string {
  return STATUS_LABELS[status] ?? status.replace(/_/g, ' ')
}

function Pill({ children, className }: { children: React.ReactNode; className?: string }) {
  return <span className={cn('rounded-full px-2 py-0.5 text-[11px] font-medium', className)}>{children}</span>
}

export function WorkQueueView({ onClose }: WorkQueueViewProps) {
  const navigate = useNavigate()
  const [items, setItems] = useState<WorkQueueItem[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  const refresh = useCallback(async () => {
    setRefreshing(true)

    try {
      const result = await getWorkQueue()
      setItems(result.items)
      setSelectedId(current => current ?? result.items[0]?.id ?? null)
    } catch (error) {
      notifyError(error, 'Failed to load work queue')
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  useRefreshHotkey(() => void refresh())

  const selected = useMemo(() => items.find(item => item.id === selectedId) ?? items[0] ?? null, [items, selectedId])

  const grouped = useMemo(
    () =>
      SECTIONS.map(section => ({
        ...section,
        items: items.filter(item => section.statuses.includes(item.status as never))
      })),
    [items]
  )

  const mutate = useCallback(
    async (action: () => Promise<WorkQueueItem>, message: string) => {
      try {
        await action()
        notify({ message })
        await refresh()
      } catch (error) {
        notifyError(error, 'Work queue action failed')
      }
    },
    [refresh]
  )

  const createSampleItem = useCallback(
    () =>
      mutate(
        () =>
          createWorkQueueItem({
            title: 'Review sample work item',
            summary: 'This is a manual test item to confirm the Work Queue is saving and displaying items.',
            detail: 'Archive or mark this done after confirming the Work Queue works.',
            source: 'manual',
            status: 'needs_review',
            priority: 'normal',
            actions: ['mark_done', 'archive', 'snooze']
          }),
        'Created sample work item'
      ),
    [mutate]
  )

  const openSelected = useCallback(() => {
    if (!selected?.source_url) {
      return
    }

    navigate(selected.source_url)
  }, [navigate, selected])

  return (
    <OverlayView headerContent={<div className="text-sm font-semibold">Work Queue</div>} onClose={onClose}>
      <div className="flex h-full min-h-0 flex-col gap-4 p-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-sm text-muted-foreground">Unified cockpit for cron failures, live sessions, desktop actions, and manual follow-up.</p>
          </div>
          <Button disabled={refreshing} onClick={() => void refresh()} size="sm" variant="outline">
            <Codicon className={cn('mr-2 size-4', refreshing && 'animate-spin')} name="refresh" />
            Refresh
          </Button>
        </div>

        {loading ? (
          <PageLoader label="Loading work queue..." />
        ) : (
          <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_minmax(320px,0.45fr)] gap-4">
            <div className="min-h-0 overflow-y-auto rounded-lg border bg-card/40">
              {items.length === 0 ? (
                <div className="flex h-full min-h-80 flex-col items-center justify-center gap-3 px-6 text-center text-muted-foreground">
                  <Codicon className="size-8" name="check-all" />
                  <div className="space-y-1">
                    <div className="text-sm font-medium text-foreground">No work items yet</div>
                    <div className="max-w-md text-xs">
                      Failed or paused cron jobs, running desktop actions, active sessions, and manual follow-ups will appear here when they need attention.
                    </div>
                  </div>
                  <Button onClick={() => void createSampleItem()} size="sm" variant="outline">
                    <Codicon className="mr-2 size-4" name="add" />
                    Create sample item
                  </Button>
                </div>
              ) : (
                <div className="divide-y">
                  {grouped.map(section => (
                    <section key={section.id}>
                      <div className="sticky top-0 z-1 flex items-center justify-between bg-background/95 px-3 py-2 backdrop-blur">
                        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{section.label}</h3>
                        <span className="text-xs text-muted-foreground">{section.items.length}</span>
                      </div>
                      {section.items.map(item => (
                        <button
                          className={cn(
                            'flex w-full flex-col gap-2 px-3 py-3 text-left transition-colors hover:bg-muted/50',
                            selected?.id === item.id && 'bg-muted'
                          )}
                          key={item.id}
                          onClick={() => setSelectedId(item.id)}
                          type="button"
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0">
                              <div className="truncate text-sm font-medium">{item.title}</div>
                              {item.summary ? <div className="mt-1 line-clamp-2 text-xs text-muted-foreground">{item.summary}</div> : null}
                            </div>
                            <Pill className={STATUS_TONE[item.status] ?? 'bg-muted text-muted-foreground'}>{statusLabel(item.status)}</Pill>
                          </div>
                          <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                            <span>{item.source}</span>
                            <span>·</span>
                            <span>{item.priority}</span>
                            <span>·</span>
                            <span>{formatWhen(item.updated_at)}</span>
                          </div>
                        </button>
                      ))}
                    </section>
                  ))}
                </div>
              )}
            </div>

            <aside className="min-h-0 overflow-hidden rounded-lg border bg-card/40">
              {selected ? (
                <div className="flex h-full min-h-0 flex-col">
                  <div className="border-b p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <h2 className="text-base font-semibold">{selected.title}</h2>
                        <p className="mt-1 text-xs text-muted-foreground">{selected.id}</p>
                      </div>
                      <Pill className={STATUS_TONE[selected.status] ?? 'bg-muted text-muted-foreground'}>{statusLabel(selected.status)}</Pill>
                    </div>
                  </div>

                  <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4 text-sm">
                    {selected.summary ? <p className="text-muted-foreground">{selected.summary}</p> : null}
                    <dl className="grid grid-cols-2 gap-3 text-xs">
                      <div><dt className="text-muted-foreground">Source</dt><dd>{selected.source}</dd></div>
                      <div><dt className="text-muted-foreground">Priority</dt><dd>{selected.priority}</dd></div>
                      <div><dt className="text-muted-foreground">Updated</dt><dd>{formatWhen(selected.updated_at)}</dd></div>
                      <div><dt className="text-muted-foreground">Due</dt><dd>{formatWhen(selected.due_at)}</dd></div>
                      {selected.actor ? <div><dt className="text-muted-foreground">Actor</dt><dd>{selected.actor}</dd></div> : null}
                      {selected.client_name ? <div><dt className="text-muted-foreground">Client</dt><dd>{selected.client_name}</dd></div> : null}
                    </dl>
                    {selected.detail ? (
                      <pre className="max-h-80 overflow-auto whitespace-pre-wrap rounded-md bg-muted p-3 text-xs text-muted-foreground">{selected.detail}</pre>
                    ) : null}
                  </div>

                  <div className="flex flex-wrap gap-2 border-t p-4">
                    {selected.source_url ? <Button onClick={openSelected} size="sm">Open</Button> : null}
                    <Button onClick={() => void mutate(() => updateWorkQueueItem(selected.id, { status: 'done' }), 'Marked done')} size="sm" variant="outline">Mark done</Button>
                    <Button onClick={() => void mutate(() => archiveWorkQueueItem(selected.id), 'Archived')} size="sm" variant="outline">Archive</Button>
                    <Button
                      onClick={() => void mutate(() => snoozeWorkQueueItem(selected.id, new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString()), 'Snoozed for 24 hours')}
                      size="sm"
                      variant="outline"
                    >
                      Snooze
                    </Button>
                  </div>
                </div>
              ) : null}
            </aside>
          </div>
        )}
      </div>
    </OverlayView>
  )
}
