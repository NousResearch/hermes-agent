import { useQuery } from '@tanstack/react-query'
import { type CSSProperties, useMemo, useState } from 'react'

import { getLatestSessionMessages, getSession } from '@/api/sessions'
import { titlebarHeightNow } from '@/app/shell/use-touch-titlebar'
import { Button } from '@/components/ui/button'
import {
  Dialog,
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
import { sessionTitle } from '@/lib/chat-runtime'
import type { SessionInfo } from '@/types/hermes'

import { agentSettingsKey, type AgentSettingsOwner } from './data'
import { useHistoryCopy } from './history-copy'
import {
  type CleanupCriteria,
  type CleanupPreview,
  DEFAULT_CLEANUP,
  deleteHistorySelection,
  type HistoryDeleteOutcome,
  type HistorySource,
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
  const { locale } = useI18n()
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
  const date = (time: number) => (time ? new Date(time * 1000).toLocaleString(locale) : c.unknown)
  const cost = info.actual_cost_usd ?? info.estimated_cost_usd

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
        showCloseButton={!busy}
        style={{ '--titlebar-height': `${titlebarHeightNow()}px` } as CSSProperties}
      >
        <DialogHeader>
          <DialogDescription>{c.readOnly}</DialogDescription>
          <DialogTitle>{sessionTitle(row)}</DialogTitle>
        </DialogHeader>
        <dl className="agent-history-metadata">
          <div>
            <dt>{c.messages}</dt>
            <dd>{info.message_count}</dd>
          </div>
          <div>
            <dt>
              {c.cost}
              {info.actual_cost_usd == null && cost != null ? ` · ${c.estimated}` : ''}
            </dt>
            <dd>
              {cost == null
                ? c.unknown
                : new Intl.NumberFormat(locale, {
                    style: 'currency',
                    currency: 'USD',
                    maximumFractionDigits: 4
                  }).format(cost)}
            </dd>
          </div>
          <div>
            <dt>{c.started}</dt>
            <dd>{date(info.started_at)}</dd>
          </div>
          <div className="agent-history-latest">
            <dt>{c.latest}</dt>
            <dd>{date(info.last_active || row.last_active || info.started_at)}</dd>
          </div>
        </dl>
        <p className="agent-history-help">{c.previewScope}</p>
        {detail.isPending && <p role="status">{c.loading}</p>}
        {detail.isError && <HistoryError error={detail.error} retry={() => void detail.refetch()} />}
        <div className="agent-history-messages">
          {messages.map(message => (
            <article className={`agent-history-message agent-history-message-${message.role}`} key={message.id}>
              <small>{message.role === 'user' ? c.you : owner.profile}</small>
              {message.timestamp && (
                <time dateTime={new Date(message.timestamp * 1000).toISOString()}>{date(message.timestamp)}</time>
              )}
              <p>{chatMessageText(message)}</p>
            </article>
          ))}
        </div>
        {Boolean(error) && <HistoryError error={error} />}
        <footer className="agent-history-preview-actions">
          <Button disabled={busy} onClick={onOpen} size="lg">
            {c.open}
          </Button>
          <Button disabled={busy} onClick={() => void archive()} size="lg" variant="secondary">
            {row.archived ? c.unarchive : c.archive}
          </Button>
          <Button disabled={busy} onClick={onDelete} size="lg" variant="destructiveOutline">
            {c.delete}
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
  current: () => boolean
  onClose: () => void
  onDeleted: (outcome: HistoryDeleteOutcome) => void
}

export function HistoryDeleteDialog({ owner, rows, current, onClose, onDeleted }: DeleteDialogProps) {
  const c = useHistoryCopy()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)

  async function confirm() {
    if (busy || !current()) {
      return
    }

    setBusy(true)
    setError(null)

    try {
      const outcome = await deleteHistorySelection(owner, rows, current, c.upgrade)

      if (current()) {
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
        showCloseButton={!busy}
      >
        <DialogHeader>
          <DialogTitle>{c.confirm}</DialogTitle>
          <DialogDescription>{c.permanent}</DialogDescription>
        </DialogHeader>
        <ul className="agent-history-delete-list">
          {rows.map(row => (
            <li key={row.id}>
              <span>{sessionTitle(row)}</span>
              <small>
                {c.messages}: {row.message_count}
              </small>
            </li>
          ))}
        </ul>
        {!rows.length && <p>{c.empty}</p>}
        {Boolean(error) && <HistoryError error={error} />}
        <DialogFooter>
          <Button disabled={busy} onClick={onClose} variant="secondary">
            {c.cancel}
          </Button>
          <Button disabled={busy || !rows.length} onClick={() => void confirm()} variant="destructive">
            {busy ? c.loading : `${c.delete} ${rows.length}`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export function HistoryCleanupDialog({ owner, current, onClose, onDeleted }: Omit<DeleteDialogProps, 'rows'>) {
  const c = useHistoryCopy()
  const [criteria, setCriteria] = useState<CleanupCriteria>(DEFAULT_CLEANUP)
  const [preview, setPreview] = useState<CleanupPreview | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)

  async function run() {
    if (busy || !current()) {
      return
    }

    setBusy(true)
    setError(null)

    try {
      if (preview) {
        const outcome = await deleteHistorySelection(owner, preview.candidates, current, c.upgrade, preview)

        if (current()) {
          outcome.running = [...new Set([...preview.running.map(row => row.id), ...outcome.running])]
          onDeleted(outcome)
          onClose()
        }
      } else {
        const result = await previewHistoryCleanup(owner, criteria, current)

        if (current()) {
          setPreview(result)
        }
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

  function change(next: CleanupCriteria) {
    setCriteria(next)
    setPreview(null)
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
        className="agent-history-dialog agent-history-cleanup-dialog"
        showCloseButton={!busy}
      >
        <DialogHeader>
          <DialogTitle>{c.cleanup}</DialogTitle>
          <DialogDescription>
            {c.nothingDeleted} {c.permanent}
          </DialogDescription>
        </DialogHeader>
        <fieldset className="agent-history-cleanup-fields" disabled={busy}>
          <HistoryChoices
            label={c.age}
            onChange={days => change({ ...criteria, days: Number(days) as CleanupCriteria['days'] })}
            options={
              [['0', c.anyTime], ...(['7', '30', '90'] as const).map(days => [days, `${days} ${c.days}`])] as Array<
                [string, string]
              >
            }
            value={String(criteria.days)}
          />
          <HistoryChoices<CleanupCriteria['messages']>
            label={c.messages}
            onChange={messages => change({ ...criteria, messages })}
            options={[
              ['any', c.any],
              ['under5', c.under5],
              ['empty', c.emptyMessages]
            ]}
            value={criteria.messages}
          />
          <HistorySourceChoices onChange={source => change({ ...criteria, source })} value={criteria.source} />
          <label className="agent-history-archive-toggle">
            {c.includeArchived}
            <Switch
              checked={criteria.includeArchived}
              onCheckedChange={value => change({ ...criteria, includeArchived: value })}
            />
          </label>
        </fieldset>
        {preview && (
          <div className="agent-history-cleanup-preview">
            <p>
              {c.delete}: {preview.candidates.length} · {c.running}: {preview.running.length}
            </p>
            <ul className="agent-history-delete-list">
              {preview.candidates.map(row => (
                <li key={row.id}>
                  <span>{sessionTitle(row)}</span>
                  <small>
                    {c.messages}: {row.message_count}
                  </small>
                </li>
              ))}
              {preview.running.map(row => (
                <li key={row.id}>
                  <span>{sessionTitle(row)}</span>
                  <small>
                    {c.running} · {c.skipped}
                  </small>
                </li>
              ))}
            </ul>
          </div>
        )}
        {Boolean(error) && <HistoryError error={error} />}
        <DialogFooter>
          <Button disabled={busy} onClick={onClose} variant="secondary">
            {c.cancel}
          </Button>
          <Button
            disabled={busy || Boolean(preview && !preview.candidates.length)}
            onClick={() => void run()}
            variant={preview ? 'destructive' : 'default'}
          >
            {busy ? c.loading : preview ? `${c.delete} ${preview.candidates.length}` : c.dryRun}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
