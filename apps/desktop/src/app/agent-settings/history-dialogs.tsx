import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { type CSSProperties, useMemo, useState } from 'react'

import { getLatestSessionMessages, getSession } from '@/api/sessions'
import { titlebarHeightNow } from '@/app/shell/use-touch-titlebar'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import { toChatMessages } from '@/lib/chat-messages/hydration'
import { chatMessageText } from '@/lib/chat-messages/parts'
import { pathLabel, sessionTitle } from '@/lib/chat-runtime'
import { AlertTriangle, X } from '@/lib/icons'
import { sessionSourceLabel } from '@/lib/session-source'
import type { SessionInfo } from '@/types/hermes'

import { agentSettingsKey, type AgentSettingsOwner } from './data'
import { useHistoryCopy } from './history-copy'
import {
  type CleanupCriteria,
  DEFAULT_CLEANUP,
  deleteHistorySelection,
  type HistoryDeleteOutcome,
  historyLastActive,
  historyRowDate,
  historyRunning,
  type HistorySource,
  historySource,
  previewHistoryCleanup
} from './history-data'

function HistoryChoices<T extends string>({
  label,
  value,
  options,
  onChange
}: {
  label: string
  value: T
  options: Array<[T, string]>
  onChange: (value: T) => void
}) {
  return (
    <fieldset className="agent-history-field">
      <legend>{label}</legend>
      <SegmentedControl
        onChange={onChange}
        options={options.map(([id, title]) => ({ id, label: title }))}
        sizing="content"
        value={value}
      />
    </fieldset>
  )
}

function HistorySourceChoices({ value, onChange }: { value: HistorySource; onChange: (value: HistorySource) => void }) {
  const c = useHistoryCopy()

  return (
    <HistoryChoices<HistorySource>
      label={c.source}
      onChange={onChange}
      options={[
        ['all', c.all],
        ['chat', c.chat],
        ['schedule', c.schedule],
        ['telegram', 'Telegram'],
        ['discord', 'Discord']
      ]}
      value={value}
    />
  )
}

export function HistoryError({ error, retry }: { error: unknown; retry?: () => void }) {
  const c = useHistoryCopy()

  return (
    <div className="agent-history-error" role="alert">
      <p>{error instanceof Error ? error.message : String(error)}</p>
      <p>{c.errorHint}</p>
      {retry && (
        <Button onClick={retry} variant="secondary">
          {c.retry}
        </Button>
      )}
    </div>
  )
}

export function HistoryPreview({
  owner,
  row,
  onClose,
  onOpen,
  onArchive,
  onDelete
}: {
  owner: AgentSettingsOwner
  row: SessionInfo
  onClose: () => void
  onOpen: () => void
  onArchive: () => Promise<void>
  onDelete: () => void
}) {
  const c = useHistoryCopy()
  const { locale, t } = useI18n()
  const sourceName = useHistorySourceName()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)

  async function archive() {
    if (busy) {
      return
    }

    setBusy(true)
    setError(null)

    try {
      await onArchive()
      onClose()
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  const detail = useQuery({
    queryKey: [...agentSettingsKey(owner), 'history-preview', row.id],
    queryFn: async () => ({
      row: await getSession(row.id, owner),
      page: await getLatestSessionMessages(row.id, owner, { passive: true })
    })
  })

  const messages = useMemo(
    () => toChatMessages(detail.data?.page.messages ?? []).filter(message => !message.hidden),
    [detail.data]
  )

  const info = detail.data?.row ?? row
  const cost = info.actual_cost_usd ?? info.estimated_cost_usd
  const started = info.started_at ? historyLastActive(info.started_at, locale) : null
  const project = info.git_repo_root || info.cwd

  const meta = [sourceName(info), project && pathLabel(project), info.git_branch, historyRowDate(info, locale)].filter(
    Boolean
  )

  return (
    <Dialog
      onOpenChange={open => {
        if (!open && !busy) {
          onClose()
        }
      }}
      open
    >
      <DialogContent
        blurBackdrop={false}
        bodyClassName="agent-history-preview-body"
        className="agent-history-preview translate-x-0 translate-y-0"
        overlayClassName="agent-history-preview-scrim"
        showCloseButton={false}
        style={{ '--titlebar-height': `${titlebarHeightNow()}px` } as CSSProperties}
      >
        {/* Header (1038:7794): eyebrow, title, "Chat · project · branch · date", the 36px close. */}
        <header className="agent-history-preview-header">
          <DialogHeader>
            <DialogDescription>{c.readOnly}</DialogDescription>
            <DialogTitle>{sessionTitle(row)}</DialogTitle>
            <p className="agent-history-preview-meta">{meta.join(' · ')}</p>
          </DialogHeader>
          <DialogClose asChild disabled={busy}>
            <Button aria-label={t.common.close} size="icon" variant="secondary">
              <X aria-hidden />
            </Button>
          </DialogClose>
        </header>
        {/* Stats (1038:7802): a card only for what the session record actually carries. */}
        <dl className="agent-history-metadata">
          <div>
            <dt>{c.messages}</dt>
            <dd>{info.message_count}</dd>
          </div>
          {cost != null && (
            <div>
              <dt>
                {c.cost}
                {info.actual_cost_usd == null ? ` · ${c.estimated}` : ''}
              </dt>
              <dd>
                {new Intl.NumberFormat(locale, { style: 'currency', currency: 'USD', maximumFractionDigits: 4 }).format(
                  cost
                )}
              </dd>
            </div>
          )}
          {started && (
            <div>
              <dt>{c.started}</dt>
              <dd title={started.absolute}>{started.label}</dd>
            </div>
          )}
        </dl>
        <section aria-busy={detail.isPending} className="agent-history-messages">
          <h3>{c.latestMessages}</h3>
          {detail.isPending && <p role="status">{c.loading}</p>}
          {detail.isError && <HistoryError error={detail.error} retry={() => void detail.refetch()} />}
          {messages.map(message => (
            <article
              className={`agent-history-message agent-history-message-${message.role}`}
              key={message.id}
              title={message.timestamp ? new Date(message.timestamp * 1000).toLocaleString(locale) : undefined}
            >
              <small>{message.role === 'user' ? c.you : owner.profile}</small>
              <p>{chatMessageText(message)}</p>
            </article>
          ))}
        </section>
        {Boolean(error) && <HistoryError error={error} />}
        <footer className="agent-history-preview-actions">
          <Button disabled={busy} onClick={onOpen} size="lg">
            {c.open}
          </Button>
          <Button disabled={busy} onClick={() => void archive()} size="lg" variant="secondary">
            <Codicon name="archive" />
            {row.archived ? c.unarchive : c.archive}
          </Button>
          <Button disabled={busy} onClick={onDelete} size="lg" variant="destructiveOutline">
            <Codicon name="trash" />
            {c.delete}…
          </Button>
        </footer>
      </DialogContent>
    </Dialog>
  )
}

export function HistoryRename({
  row,
  onClose,
  onSave
}: {
  row: SessionInfo
  onClose: () => void
  onSave: (title: string) => Promise<void>
}) {
  const c = useHistoryCopy()
  const [title, setTitle] = useState(sessionTitle(row))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)

  return (
    <Dialog
      onOpenChange={open => {
        if (!open && !busy) {
          onClose()
        }
      }}
      open
    >
      <DialogContent
        blurBackdrop={false}
        bodyClassName="agent-history-dialog-body"
        className="agent-history-dialog"
        showCloseButton={!busy}
      >
        <DialogHeader>
          <DialogTitle>{c.rename}</DialogTitle>
          <DialogDescription>{sessionTitle(row)}</DialogDescription>
        </DialogHeader>
        <form
          onSubmit={event => {
            event.preventDefault()

            if (busy || !title.trim()) {
              return
            }

            setBusy(true)
            setError(null)
            void onSave(title.trim())
              .then(onClose)
              .catch(setError)
              .finally(() => setBusy(false))
          }}
        >
          <Input
            aria-label={c.rename}
            autoFocus
            disabled={busy}
            onChange={event => setTitle(event.target.value)}
            required
            value={title}
          />
          {Boolean(error) && <HistoryError error={error} />}
          <DialogFooter>
            <Button disabled={busy} onClick={onClose} type="button" variant="secondary">
              {c.cancel}
            </Button>
            <Button disabled={busy || !title.trim()} type="submit">
              {busy ? c.loading : c.save}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

interface DeleteDialogProps {
  owner: AgentSettingsOwner
  rows: SessionInfo[]
  /** Live running set: running rows are named in a warning and never sent for deletion. */
  running: Set<string>
  current: () => boolean
  onClose: () => void
  onDeleted: (outcome: HistoryDeleteOutcome) => void
}

/** "Chat", "Schedule", or the messaging platform's name. */
export function useHistorySourceName() {
  const c = useHistoryCopy()
  const labels: Record<string, string> = { chat: c.chat, schedule: c.schedule }

  return (row: SessionInfo) => labels[historySource(row)] ?? sessionSourceLabel(row.source) ?? c.other
}

/** Confirmation rows (878:8022): title over "Chat · Nov 1 · 0 messages"; a skipped row is dimmed
 *  and chipped. */
function HistoryItemList({ rows, skipped = [] }: { rows: SessionInfo[]; skipped?: SessionInfo[] }) {
  const c = useHistoryCopy()
  const { locale } = useI18n()
  const sourceName = useHistorySourceName()

  const item = (row: SessionInfo, isSkipped: boolean) => (
    <li data-skipped={isSkipped || undefined} key={row.id}>
      <span>
        <span>{sessionTitle(row)}</span>
        <small>
          {[sourceName(row), historyRowDate(row, locale), c.messageCount(row.message_count)]
            .filter(Boolean)
            .join(' · ')}
        </small>
      </span>
      {isSkipped && <em>{c.runningSkipped}</em>}
    </li>
  )

  return (
    <ul className="agent-history-delete-list">
      {rows.map(row => item(row, false))}
      {skipped.map(row => item(row, true))}
    </ul>
  )
}

/** The amber callout (878:8032) naming what a running item makes the delete skip. */
function HistorySkippedCallout({ rows }: { rows: SessionInfo[] }) {
  const c = useHistoryCopy()

  if (!rows.length) {
    return null
  }

  return (
    <p className="agent-history-skipped" role="status">
      <AlertTriangle aria-hidden />
      {rows.length === 1 ? c.skippedOne(sessionTitle(rows[0]!)) : c.skippedMany(rows.length)}
    </p>
  )
}

export function HistoryDeleteDialog({ owner, rows, running, current, onClose, onDeleted }: DeleteDialogProps) {
  const c = useHistoryCopy()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)

  // Split once: the list must not reshuffle under the pointer while the dialog is open.
  const [{ deletable, skipped }] = useState(() => ({
    deletable: rows.filter(row => !historyRunning(row, running)),
    skipped: rows.filter(row => historyRunning(row, running))
  }))

  async function confirm() {
    if (busy || !current()) {
      return
    }

    setBusy(true)
    setError(null)

    try {
      const outcome = await deleteHistorySelection(owner, deletable, current, c.upgrade)

      if (current()) {
        outcome.running = [...new Set([...skipped.map(row => row.id), ...outcome.running])]
        onDeleted(outcome)
        onClose()
      }
    } catch (err) {
      if (current()) {
        setError(err)
      }
    } finally {
      if (current()) {
        setBusy(false)
      }
    }
  }

  return (
    <Dialog
      onOpenChange={open => {
        if (!open && !busy) {
          onClose()
        }
      }}
      open
    >
      <DialogContent
        blurBackdrop={false}
        bodyClassName="agent-history-dialog-body"
        className="agent-history-dialog"
        showCloseButton={false}
      >
        <DialogHeader>
          <DialogTitle>{c.deleteTitle(deletable.length)}</DialogTitle>
          <DialogDescription>{c.deleteDescription}</DialogDescription>
        </DialogHeader>
        {deletable.length > 0 && <HistoryItemList rows={deletable} />}
        <HistorySkippedCallout rows={skipped} />
        {Boolean(error) && <HistoryError error={error} />}
        <DialogFooter>
          <Button disabled={busy} onClick={onClose} variant="secondary">
            {c.cancel}
          </Button>
          <Button disabled={busy || !deletable.length} onClick={() => void confirm()} variant="destructive">
            {busy ? c.loading : c.deleteItems(deletable.length)}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

interface CleanupDialogProps extends Omit<DeleteDialogProps, 'rows' | 'running'> {
  /** Where the options start: the summary's "empty chats · Clean up" link asks for empty chats. */
  initial?: Partial<CleanupCriteria>
}

/** One dialog (878:9901): the options drive a live dry run, and the confirm deletes exactly the
 *  previewed rows (re-validated per row against the same criteria by `deleteHistorySelection`). */
export function HistoryCleanupDialog({ owner, current, onClose, onDeleted, initial }: CleanupDialogProps) {
  const c = useHistoryCopy()
  const [criteria, setCriteria] = useState<CleanupCriteria>(() => ({ ...DEFAULT_CLEANUP, ...initial }))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)

  const dryRun = useQuery({
    queryKey: [...agentSettingsKey(owner), 'history-cleanup', criteria],
    queryFn: () => previewHistoryCleanup(owner, criteria, current),
    // Keep the last list on screen while the next options are counted, so the dialog never jumps.
    placeholderData: keepPreviousData,
    gcTime: 0
  })

  // A placeholder belongs to the previous options; deleting from it would widen consent.
  const preview = dryRun.isPlaceholderData ? undefined : dryRun.data
  const shown = dryRun.data

  async function run() {
    if (busy || !preview || !current()) {
      return
    }

    setBusy(true)
    setError(null)

    try {
      const outcome = await deleteHistorySelection(owner, preview.candidates, current, c.upgrade, preview)

      if (current()) {
        outcome.running = [...new Set([...preview.running.map(row => row.id), ...outcome.running])]
        onDeleted(outcome)
        onClose()
      }
    } catch (err) {
      if (current()) {
        setError(err)
      }
    } finally {
      if (current()) {
        setBusy(false)
      }
    }
  }

  const change = (next: Partial<CleanupCriteria>) => setCriteria(previous => ({ ...previous, ...next }))

  return (
    <Dialog
      onOpenChange={open => {
        if (!open && !busy) {
          onClose()
        }
      }}
      open
    >
      <DialogContent
        blurBackdrop={false}
        bodyClassName="agent-history-dialog-body"
        className="agent-history-dialog agent-history-cleanup-dialog"
        showCloseButton={false}
      >
        <DialogHeader>
          <DialogTitle>{c.cleanupTitle}</DialogTitle>
          <DialogDescription>{c.cleanupDescription}</DialogDescription>
        </DialogHeader>
        <fieldset className="agent-history-cleanup-fields" disabled={busy}>
          <HistoryChoices
            label={c.age}
            onChange={days => change({ days: Number(days) as CleanupCriteria['days'] })}
            options={
              [['0', c.anyTime], ...([7, 30, 90] as const).map(days => [String(days), c.olderThan(days)])] as Array<
                [string, string]
              >
            }
            value={String(criteria.days)}
          />
          <HistoryChoices<CleanupCriteria['messages']>
            label={c.messages}
            onChange={messages => change({ messages })}
            options={[
              ['any', c.any],
              ['under5', c.under5],
              ['empty', c.emptyOnly]
            ]}
            value={criteria.messages}
          />
          <HistorySourceChoices onChange={source => change({ source })} value={criteria.source} />
          <label className="agent-history-archive-toggle">
            {c.includeArchived}
            <Switch checked={criteria.includeArchived} onCheckedChange={value => change({ includeArchived: value })} />
          </label>
        </fieldset>
        <section aria-busy={dryRun.isFetching} aria-live="polite" className="agent-history-cleanup-preview">
          <header>
            <h3>{shown ? c.willDelete(shown.candidates.length) : c.loading}</h3>
            <span>{c.previewNote}</span>
          </header>
          {shown && shown.candidates.length + shown.running.length > 0 && (
            <HistoryItemList rows={shown.candidates} skipped={shown.running} />
          )}
          {shown && !shown.candidates.length && !shown.running.length && (
            <p className="agent-history-help">{c.nothingMatches}</p>
          )}
        </section>
        {dryRun.isError && <HistoryError error={dryRun.error} retry={() => void dryRun.refetch()} />}
        {Boolean(error) && <HistoryError error={error} />}
        <DialogFooter>
          <Button disabled={busy} onClick={onClose} variant="secondary">
            {c.cancel}
          </Button>
          <Button disabled={busy || !preview?.candidates.length} onClick={() => void run()} variant="destructive">
            {busy ? c.loading : c.deleteItems(preview?.candidates.length ?? shown?.candidates.length ?? 0)}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
