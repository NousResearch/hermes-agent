import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { pinSession, unpinSession } from '@/store/layout'
import { sessionPinId } from '@/store/session'
import { $sidebarSelection, clearSidebarSelection } from '@/store/sidebar-selection'
import type { SessionInfo } from '@/types/hermes'

import { BulkRuntimeTextDialog, type BulkRuntimeTextMode } from './bulk-runtime-text-dialog'

interface SelectionActionBarProps {
  /** The owning section's rows — resolves selected ids to pin ids/profiles. */
  sessions: SessionInfo[]
  onArchiveSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onRestoreSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onDeleteSessions?: (sessionIds: string[]) => Promise<unknown> | void
  onPromptSessions?: (sessionIds: string[], text: string) => Promise<unknown> | void
  onSteerSessions?: (sessionIds: string[], text: string) => Promise<unknown> | void
  onHaltSessions?: (sessionIds: string[]) => Promise<unknown> | void
}

type PendingAction = 'archive' | 'delete' | 'halt' | 'pin' | 'prompt' | 'restore' | 'steer' | null

/** Selection-mode header for a sidebar section: while rows in the section are
 * selected, this REPLACES the section's own header row — the live count and
 * the bulk verbs sit directly above the checked rows, where the user is
 * already looking (the earlier bottom-of-sidebar placement was undiscoverable).
 * Mounted only while a selection exists, so it also owns the Esc-to-clear
 * binding. */
export function SelectionActionBar({
  sessions,
  onArchiveSessions,
  onRestoreSessions,
  onDeleteSessions,
  onPromptSessions,
  onSteerSessions,
  onHaltSessions
}: SelectionActionBarProps) {
  const { t } = useI18n()
  const s = t.sidebar.bulk
  const selection = useStore($sidebarSelection)
  const [pending, setPending] = useState<PendingAction>(null)
  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false)
  const [runtimeTextMode, setRuntimeTextMode] = useState<BulkRuntimeTextMode | null>(null)

  const count = selection.ids.length
  const active = selection.section !== null && count > 0

  const sessionsById = useMemo(() => new Map(sessions.map(session => [session.id, session])), [sessions])

  useEffect(() => {
    if (!active) {
      return
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        clearSidebarSelection()
      }
    }

    window.addEventListener('keydown', onKeyDown)

    return () => window.removeEventListener('keydown', onKeyDown)
  }, [active])

  // The confirm dialog can't outlive the selection it asks about.
  useEffect(() => {
    if (!active) {
      setConfirmDeleteOpen(false)
      setRuntimeTextMode(null)
    }
  }, [active])

  if (!active) {
    return null
  }

  const isArchivedSection = selection.section === 'archived'
  const isPinnedSection = selection.section === 'pinned'

  const runBulk = async (action: Exclude<PendingAction, null>, run: () => Promise<unknown> | void) => {
    if (pending) {
      return
    }

    setPending(action)

    try {
      await run()
      clearSidebarSelection()
    } finally {
      setPending(null)
    }
  }

  const togglePins = () => {
    triggerHaptic('selection')

    void runBulk('pin', () => {
      for (const id of selection.ids) {
        const session = sessionsById.get(id)
        const pinId = session ? sessionPinId(session) : id

        if (isPinnedSection) {
          unpinSession(pinId)
        } else {
          pinSession(pinId)
        }
      }
    })
  }

  const archiveSelected = () => {
    triggerHaptic('selection')
    void runBulk('archive', () => onArchiveSessions?.([...selection.ids]))
  }

  const restoreSelected = () => {
    triggerHaptic('selection')
    void runBulk('restore', () => onRestoreSessions?.([...selection.ids]))
  }

  const deleteSelected = () => {
    triggerHaptic('warning')
    setConfirmDeleteOpen(false)
    void runBulk('delete', () => onDeleteSessions?.([...selection.ids]))
  }

  const haltSelected = () => {
    triggerHaptic('warning')
    void runBulk('halt', () => onHaltSessions?.([...selection.ids]))
  }

  const submitRuntimeText = (mode: BulkRuntimeTextMode, text: string) => {
    setRuntimeTextMode(null)
    triggerHaptic('submit')

    void runBulk(mode, () =>
      mode === 'prompt' ? onPromptSessions?.([...selection.ids], text) : onSteerSessions?.([...selection.ids], text)
    )
  }

  const actionButton = (
    label: string,
    icon: string,
    onClick: () => void,
    action: Exclude<PendingAction, null>,
    destructive = false,
    disabled = false
  ) => (
    <Tip key={label} label={label}>
      <Button
        aria-label={label}
        className={
          destructive
            ? 'size-5 text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-destructive'
            : 'size-5 text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground'
        }
        disabled={pending !== null || disabled}
        onClick={onClick}
        size="icon-xs"
        variant="ghost"
      >
        <Codicon name={pending === action ? 'loading' : icon} size="0.8125rem" spinning={pending === action} />
      </Button>
    </Tip>
  )

  return (
    // Same geometry contract as SidebarSectionHeader (pl-2 + w-3.5 slot +
    // gap-1.5 lead, min-h-[1.875rem], pr-0.5 trailing) so swapping the header
    // for this bar moves nothing: the count sits on the section-label edge and
    // the verbs land where the header actions were.
    <div className="flex min-h-[1.875rem] shrink-0 items-center gap-0.5 pb-1 pr-0.5 pt-1.5" data-selection-bar>
      <span className="flex min-w-0 flex-1 items-center gap-1.5 pl-2">
        <span aria-hidden="true" className="grid w-3.5 shrink-0 place-items-center text-(--ui-text-secondary)">
          <Codicon name="check" size="0.75rem" />
        </span>
        <span className="min-w-0 truncate text-[0.6875rem] font-semibold uppercase tracking-wide text-(--ui-text-secondary)">
          {s.selectedCount(count)}
        </span>
      </span>
      {!isArchivedSection &&
        actionButton(
          s.promptCount(count),
          'arrow-up',
          () => setRuntimeTextMode('prompt'),
          'prompt',
          false,
          !onPromptSessions
        )}
      {!isArchivedSection &&
        actionButton(
          s.steerCount(count),
          'comment-discussion',
          () => setRuntimeTextMode('steer'),
          'steer',
          false,
          !onSteerSessions
        )}
      {!isArchivedSection &&
        actionButton(s.haltCount(count), 'debug-stop', haltSelected, 'halt', true, !onHaltSessions)}
      {!isArchivedSection &&
        actionButton(isPinnedSection ? s.unpin : s.pin, isPinnedSection ? 'pinned' : 'pin', togglePins, 'pin')}
      {!isArchivedSection && actionButton(s.archive, 'archive', archiveSelected, 'archive')}
      {isArchivedSection && actionButton(s.restore, 'history', restoreSelected, 'restore')}
      {actionButton(s.delete, 'trash', () => setConfirmDeleteOpen(true), 'delete', true)}
      <Tip label={s.clearSelection}>
        <Button
          aria-label={s.clearSelection}
          className="size-5 text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground"
          disabled={pending !== null}
          onClick={() => clearSidebarSelection()}
          size="icon-xs"
          variant="ghost"
        >
          <Codicon name="close" size="0.8125rem" />
        </Button>
      </Tip>
      <Dialog onOpenChange={setConfirmDeleteOpen} open={confirmDeleteOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{s.deleteDialogTitle(count)}</DialogTitle>
            <DialogDescription>{s.deleteDialogDesc}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button onClick={() => setConfirmDeleteOpen(false)} type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button onClick={deleteSelected} type="button" variant="destructive">
              {s.deleteConfirm}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <BulkRuntimeTextDialog
        count={count}
        mode={runtimeTextMode}
        onOpenChange={open => setRuntimeTextMode(open ? runtimeTextMode : null)}
        onSubmit={submitRuntimeText}
        pending={pending !== null}
      />
    </div>
  )
}
