import { useState } from 'react'

import { PanelListRow, type PanelMenuItem, PanelRowMenu } from '@/app/overlays/panel'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import type { SessionInfo } from '@/types/hermes'

interface CronRunRowProps {
  active?: boolean
  onDelete: (sessionId: string, profile?: string) => Promise<boolean>
  onDeleted: (sessionId: string) => void
  onOpen: (sessionId: string) => void
  run: SessionInfo
  time: string
  variant: 'detail' | 'sidebar'
}

function runTitle(run: SessionInfo): string {
  return run.title?.trim() || run.preview?.trim() || run.id
}

/**
 * One cron-run row for both run-history surfaces. Cron runs are stored sessions,
 * so deletion deliberately goes through the app's session action instead of
 * issuing a second REST path here. The caller removes the row only after that
 * action reports success.
 */
export function CronRunRow({ active = false, onDelete, onDeleted, onOpen, run, time, variant }: CronRunRowProps) {
  const { t } = useI18n()
  const copy = t.sidebar.row
  const [deleteOpen, setDeleteOpen] = useState(false)
  const title = runTitle(run)

  const menuItems: PanelMenuItem[] = [
    {
      icon: 'trash',
      label: t.common.delete,
      onSelect: () => setDeleteOpen(true),
      tone: 'danger'
    }
  ]

  const confirmDelete = async () => {
    const deleted = await onDelete(run.id, run.profile ?? undefined)

    if (!deleted) {
      throw new Error(t.desktop.deleteFailed)
    }

    onDeleted(run.id)
  }

  return (
    <>
      {variant === 'detail' ? (
        <PanelListRow
          active={false}
          menuItems={menuItems}
          menuLabel={t.sidebar.row.sessionActions}
          meta={time}
          onSelect={() => onOpen(run.id)}
          rowKey={run.id}
          title={title}
        />
      ) : (
        <div
          className={cn(
            'group/row flex min-w-0 items-center rounded-md focus-within:ring-2 focus-within:ring-ring/40',
            active ? 'bg-(--ui-row-active-background) text-foreground' : 'text-(--ui-text-secondary)'
          )}
        >
          <button
            aria-label={`${title} — ${time}`}
            className="min-w-0 flex-1 truncate rounded-md px-1.5 py-0.5 text-left text-[0.6875rem] tabular-nums hover:bg-(--chrome-action-hover) hover:text-foreground focus-visible:bg-(--chrome-action-hover) focus-visible:text-foreground focus-visible:outline-none"
            onClick={() => onOpen(run.id)}
            type="button"
          >
            {time}
          </button>
          <div className="shrink-0 pr-0.5">
            <PanelRowMenu items={menuItems} label={t.sidebar.row.sessionActions} />
          </div>
        </div>
      )}

      <ConfirmDialog
        busyLabel={copy.deleting}
        confirmLabel={t.common.delete}
        description={copy.deleteDesc(title)}
        destructive
        doneLabel={copy.deleted}
        onClose={() => setDeleteOpen(false)}
        onConfirm={confirmDelete}
        open={deleteOpen}
        title={copy.deleteTitle}
      />
    </>
  )
}
