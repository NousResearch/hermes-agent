import type * as React from 'react'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from '@/components/ui/context-menu'
import { writeClipboardText } from '@/components/ui/copy-button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { renameSession } from '@/hermes'
import { useI18n } from '@/i18n'
import { copyCloudChannelId, inviteCloudChannelMember, shareSessionToCloud } from '@/lib/cloud-share'
import { triggerHaptic } from '@/lib/haptics'
import { exportSession } from '@/lib/session-export'
import { notify, notifyError } from '@/store/notifications'
import { setSessions } from '@/store/session'
import { canOpenSessionWindow, openSessionInNewWindow } from '@/store/windows'

interface SessionActions {
  sessionId: string
  title: string
  pinned?: boolean
  profile?: string
  /** Archived-section rows: swap Archive→Restore and drop the pin/rename
   * entries that don't apply to a row living in cold storage. */
  archived?: boolean
  onPin?: () => void
  onArchive?: () => void
  onRestore?: () => void
  onDelete?: () => void
  /** Adds this row to its section's multi-select. */
  onSelect?: () => void
}

type MenuItem = typeof DropdownMenuItem | typeof ContextMenuItem

interface ItemSpec {
  className?: string
  disabled: boolean
  icon: string
  label: string
  onSelect: (event: Event) => void
  variant?: 'destructive'
}

function useSessionActions({
  sessionId,
  title,
  pinned = false,
  profile,
  archived = false,
  onPin,
  onArchive,
  onRestore,
  onDelete,
  onSelect
}: SessionActions) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const [renameOpen, setRenameOpen] = useState(false)
  const [inviteOpen, setInviteOpen] = useState(false)

  const selectItem: ItemSpec[] = onSelect
    ? [
        {
          disabled: !sessionId,
          icon: 'checklist',
          label: r.select,
          onSelect: () => {
            onSelect()
          }
        }
      ]
    : []

  const deleteItem: ItemSpec = {
    className: 'text-destructive focus:text-destructive',
    disabled: !onDelete,
    icon: 'trash',
    label: t.common.delete,
    onSelect: () => {
      triggerHaptic('warning')
      onDelete?.()
    },
    variant: 'destructive'
  }

  const copyIdItem: ItemSpec = {
    disabled: !sessionId,
    icon: 'copy',
    label: r.copyId,
    onSelect: event => {
      event.preventDefault()
      triggerHaptic('selection')
      void writeClipboardText(sessionId).catch(err => notifyError(err, r.copyIdFailed))
    }
  }

  const shareToCloudItem: ItemSpec = {
    disabled: !sessionId,
    icon: 'cloud-upload',
    label: r.shareToCloud,
    onSelect: () => {
      triggerHaptic('selection')
      void shareSessionToCloud(sessionId)
    }
  }

  const copyCloudIdItem: ItemSpec = {
    disabled: !sessionId,
    icon: 'copy',
    label: r.copyCloudId,
    onSelect: event => {
      event.preventDefault()
      triggerHaptic('selection')
      void copyCloudChannelId(sessionId)
    }
  }

  const inviteToCloudItem: ItemSpec = {
    disabled: !sessionId,
    icon: 'mail',
    label: r.inviteToCloud,
    onSelect: () => {
      triggerHaptic('selection')
      setInviteOpen(true)
    }
  }

  const exportItem: ItemSpec = {
    disabled: !sessionId,
    icon: 'cloud-download',
    label: r.export,
    onSelect: () => {
      triggerHaptic('selection')
      void exportSession(sessionId, { title })
    }
  }

  const items: ItemSpec[] = archived
    ? [
        ...selectItem,
        {
          disabled: !onRestore,
          icon: 'history',
          label: r.restore,
          onSelect: () => {
            triggerHaptic('selection')
            onRestore?.()
          }
        },
        copyIdItem,
        exportItem,
        deleteItem
      ]
    : [
        ...selectItem,
        {
          disabled: !onPin,
          icon: 'pin',
          label: pinned ? r.unpin : r.pin,
          onSelect: () => {
            triggerHaptic('selection')
            onPin?.()
          }
        },
        copyIdItem,
        ...(canOpenSessionWindow()
          ? [
              {
                disabled: !sessionId,
                icon: 'link-external',
                label: r.newWindow,
                onSelect: () => {
                  triggerHaptic('selection')
                  void openSessionInNewWindow(sessionId)
                }
              }
            ]
          : []),
        shareToCloudItem,
        copyCloudIdItem,
        inviteToCloudItem,
        exportItem,
        {
          disabled: !sessionId,
          icon: 'edit',
          label: r.rename,
          onSelect: () => {
            triggerHaptic('selection')
            setRenameOpen(true)
          }
        },
        {
          disabled: !onArchive,
          icon: 'archive',
          label: r.archive,
          onSelect: () => {
            triggerHaptic('selection')
            onArchive?.()
          }
        },
        deleteItem
      ]

  const renderItems = (Item: MenuItem) =>
    items.map(({ className, disabled, icon, label, onSelect, variant }) => (
      <Item className={className} disabled={disabled} key={label} onSelect={onSelect} variant={variant}>
        <Codicon name={icon} size="0.875rem" />
        <span>{label}</span>
      </Item>
    ))

  const renameDialog = (
    <RenameSessionDialog
      currentTitle={title}
      onOpenChange={setRenameOpen}
      open={renameOpen}
      profile={profile}
      sessionId={sessionId}
    />
  )

  const inviteDialog = (
    <InviteCloudDialog onOpenChange={setInviteOpen} open={inviteOpen} sessionId={sessionId} />
  )

  return { inviteDialog, renameDialog, renderItems }
}

interface SessionActionsMenuProps
  extends SessionActions, Pick<React.ComponentProps<typeof DropdownMenuContent>, 'align' | 'sideOffset'> {
  children: React.ReactNode
}

export function SessionActionsMenu({ children, align = 'end', sideOffset = 6, ...actions }: SessionActionsMenuProps) {
  const { t } = useI18n()
  const { inviteDialog, renameDialog, renderItems } = useSessionActions(actions)

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>{children}</DropdownMenuTrigger>
        <DropdownMenuContent
          align={align}
          aria-label={t.sidebar.row.actionsFor(actions.title)}
          className="w-40"
          sideOffset={sideOffset}
        >
          {renderItems(DropdownMenuItem)}
        </DropdownMenuContent>
      </DropdownMenu>
      {inviteDialog}
      {renameDialog}
    </>
  )
}

interface SessionContextMenuProps extends SessionActions {
  children: React.ReactNode
}

export function SessionContextMenu({ children, ...actions }: SessionContextMenuProps) {
  const { t } = useI18n()
  const { inviteDialog, renameDialog, renderItems } = useSessionActions(actions)

  return (
    <>
      <ContextMenu>
        <ContextMenuTrigger asChild>{children}</ContextMenuTrigger>
        <ContextMenuContent aria-label={t.sidebar.row.actionsFor(actions.title)} className="w-40">
          {renderItems(ContextMenuItem)}
        </ContextMenuContent>
      </ContextMenu>
      {inviteDialog}
      {renameDialog}
    </>
  )
}

interface InviteCloudDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  sessionId: string
}

function InviteCloudDialog({ open, onOpenChange, sessionId }: InviteCloudDialogProps) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const [email, setEmail] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setEmail('')
      window.setTimeout(() => inputRef.current?.focus(), 0)
    }
  }, [open])

  const submit = async () => {
    if (!sessionId || submitting) {
      return
    }

    setSubmitting(true)

    try {
      const ok = await inviteCloudChannelMember(sessionId, email)
      if (ok) {
        onOpenChange(false)
      }
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{r.inviteCloudTitle}</DialogTitle>
          <DialogDescription>{r.inviteCloudDesc}</DialogDescription>
        </DialogHeader>
        <Input
          autoFocus
          disabled={submitting}
          onChange={event => setEmail(event.target.value)}
          onKeyDown={event => {
            if (event.key === 'Enter') {
              event.preventDefault()
              void submit()
            } else if (event.key === 'Escape') {
              onOpenChange(false)
            }
          }}
          placeholder={r.inviteEmailPlaceholder}
          ref={inputRef}
          type="email"
          value={email}
        />
        <DialogFooter>
          <Button disabled={submitting} onClick={() => onOpenChange(false)} type="button" variant="ghost">
            {t.common.cancel}
          </Button>
          <Button disabled={submitting} onClick={() => void submit()} type="button">
            {r.inviteCreate}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

interface RenameSessionDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  sessionId: string
  currentTitle: string
  profile?: string
}

function RenameSessionDialog({ open, onOpenChange, sessionId, currentTitle, profile }: RenameSessionDialogProps) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const [value, setValue] = useState(currentTitle)
  const [submitting, setSubmitting] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setValue(currentTitle)
      window.setTimeout(() => inputRef.current?.select(), 0)
    }
  }, [currentTitle, open])

  const submit = async () => {
    const next = value.trim()

    if (!sessionId || submitting) {
      return
    }

    if (next === currentTitle.trim()) {
      onOpenChange(false)

      return
    }

    setSubmitting(true)

    try {
      const result = await renameSession(sessionId, next, profile)
      const finalTitle = result.title || next || ''
      setSessions(prev => prev.map(s => (s.id === sessionId ? { ...s, title: finalTitle || null } : s)))
      notify({ durationMs: 2_000, kind: 'success', message: r.renamed })
      onOpenChange(false)
    } catch (err) {
      notifyError(err, r.renameFailed)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{r.renameTitle}</DialogTitle>
          <DialogDescription>{r.renameDesc}</DialogDescription>
        </DialogHeader>
        <Input
          autoFocus
          disabled={submitting}
          onChange={event => setValue(event.target.value)}
          onKeyDown={event => {
            if (event.key === 'Enter') {
              event.preventDefault()
              void submit()
            } else if (event.key === 'Escape') {
              onOpenChange(false)
            }
          }}
          placeholder={r.untitledPlaceholder}
          ref={inputRef}
          value={value}
        />
        <DialogFooter>
          <Button disabled={submitting} onClick={() => onOpenChange(false)} type="button" variant="ghost">
            {t.common.cancel}
          </Button>
          <Button disabled={submitting} onClick={() => void submit()} type="button">
            {t.common.save}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
