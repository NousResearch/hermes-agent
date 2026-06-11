import type * as React from 'react'
import { useCallback, useEffect, useRef, useState } from 'react'

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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tip } from '@/components/ui/tooltip'
import { renameSession } from '@/hermes'
import { useI18n } from '@/i18n'
import {
  type CloudChannelMember,
  type CloudChannelPermission,
  type CloudMembersResult,
  copyCloudChannelId,
  deleteCloudChannel,
  inviteCloudChannelMember,
  loadCloudChannelMembers,
  removeCloudChannelMember,
  setCloudChannelMemberPermission,
  shareSessionToCloud
} from '@/lib/cloud-share'
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
  const [membersOpen, setMembersOpen] = useState(false)
  const [deleteCloudOpen, setDeleteCloudOpen] = useState(false)

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

  const cloudMembersItem: ItemSpec = {
    disabled: !sessionId,
    icon: 'organization',
    label: r.cloudMembers,
    onSelect: () => {
      triggerHaptic('selection')
      setMembersOpen(true)
    }
  }

  const deleteCloudItem: ItemSpec = {
    className: 'text-destructive focus:text-destructive',
    disabled: !sessionId,
    icon: 'trash',
    label: r.deleteCloudChannel,
    onSelect: () => {
      triggerHaptic('warning')
      setDeleteCloudOpen(true)
    },
    variant: 'destructive'
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
        {
          disabled: !sessionId,
          icon: 'edit',
          label: r.rename,
          onSelect: () => {
            triggerHaptic('selection')
            setRenameOpen(true)
          }
        },
        copyIdItem,
        copyCloudIdItem,
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
        exportItem,
        shareToCloudItem,
        inviteToCloudItem,
        cloudMembersItem,
        deleteCloudItem,
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

  const membersDialog = (
    <CloudMembersDialog onOpenChange={setMembersOpen} open={membersOpen} sessionId={sessionId} />
  )

  const deleteCloudDialog = (
    <DeleteCloudChannelDialog onOpenChange={setDeleteCloudOpen} open={deleteCloudOpen} sessionId={sessionId} />
  )

  return { deleteCloudDialog, inviteDialog, membersDialog, renameDialog, renderItems }
}

interface SessionActionsMenuProps
  extends SessionActions, Pick<React.ComponentProps<typeof DropdownMenuContent>, 'align' | 'sideOffset'> {
  children: React.ReactNode
}

export function SessionActionsMenu({ children, align = 'end', sideOffset = 6, ...actions }: SessionActionsMenuProps) {
  const { t } = useI18n()
  const { deleteCloudDialog, inviteDialog, membersDialog, renameDialog, renderItems } = useSessionActions(actions)

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>{children}</DropdownMenuTrigger>
        <DropdownMenuContent
          align={align}
          aria-label={t.sidebar.row.actionsFor(actions.title)}
          className="w-48"
          sideOffset={sideOffset}
        >
          {renderItems(DropdownMenuItem)}
        </DropdownMenuContent>
      </DropdownMenu>
      {deleteCloudDialog}
      {inviteDialog}
      {membersDialog}
      {renameDialog}
    </>
  )
}

interface SessionContextMenuProps extends SessionActions {
  children: React.ReactNode
}

export function SessionContextMenu({ children, ...actions }: SessionContextMenuProps) {
  const { t } = useI18n()
  const { deleteCloudDialog, inviteDialog, membersDialog, renameDialog, renderItems } = useSessionActions(actions)

  return (
    <>
      <ContextMenu>
        <ContextMenuTrigger asChild>{children}</ContextMenuTrigger>
        <ContextMenuContent aria-label={t.sidebar.row.actionsFor(actions.title)} className="w-48">
          {renderItems(ContextMenuItem)}
        </ContextMenuContent>
      </ContextMenu>
      {deleteCloudDialog}
      {inviteDialog}
      {membersDialog}
      {renameDialog}
    </>
  )
}

interface DeleteCloudChannelDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  sessionId: string
}

function DeleteCloudChannelDialog({ open, onOpenChange, sessionId }: DeleteCloudChannelDialogProps) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const [deleting, setDeleting] = useState(false)

  const confirmDelete = async () => {
    if (deleting) {
      return
    }

    setDeleting(true)

    try {
      if (await deleteCloudChannel(sessionId)) {
        onOpenChange(false)
      }
    } finally {
      setDeleting(false)
    }
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{r.deleteCloudTitle}</DialogTitle>
          <DialogDescription>{r.deleteCloudDesc}</DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button disabled={deleting} onClick={() => onOpenChange(false)} type="button" variant="ghost">
            {t.common.cancel}
          </Button>
          <Button disabled={deleting} onClick={() => void confirmDelete()} type="button" variant="destructive">
            {deleting ? r.deleteCloudDeleting : r.deleteCloudConfirm}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

interface CloudMembersDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  sessionId: string
}

const CLOUD_MEMBER_PERMISSIONS: CloudChannelPermission[] = ['read', 'post', 'admin']

const cloudMemberPermission = (permission: string | null | undefined): CloudChannelPermission =>
  permission === 'post' || permission === 'admin' ? permission : 'read'

function CloudMembersDialog({ open, onOpenChange, sessionId }: CloudMembersDialogProps) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const [loading, setLoading] = useState(false)
  const [pendingMemberId, setPendingMemberId] = useState<string | null>(null)
  const [result, setResult] = useState<CloudMembersResult | null>(null)

  const refreshMembers = useCallback(async () => {
    setLoading(true)

    try {
      setResult(await loadCloudChannelMembers(sessionId))
    } finally {
      setLoading(false)
    }
  }, [sessionId])

  useEffect(() => {
    if (open) {
      void refreshMembers()
    }
  }, [open, refreshMembers])

  const updatePermission = async (member: CloudChannelMember, permission: CloudChannelPermission) => {
    const current = cloudMemberPermission(member.permission)

    if (permission === current || pendingMemberId) {
      return
    }

    setPendingMemberId(`permission:${member.account_id}`)

    try {
      if (await setCloudChannelMemberPermission(sessionId, member.account_id, permission)) {
        await refreshMembers()
      }
    } finally {
      setPendingMemberId(null)
    }
  }

  const removeMember = async (member: CloudChannelMember) => {
    if (pendingMemberId) {
      return
    }

    const label = member.display_name || member.email || member.account_id

    if (!window.confirm(r.cloudMembersRevokeConfirm(label))) {
      return
    }

    setPendingMemberId(`remove:${member.account_id}`)

    try {
      if (await removeCloudChannelMember(sessionId, member.account_id)) {
        await refreshMembers()
      }
    } finally {
      setPendingMemberId(null)
    }
  }

  const members = result?.members ?? []
  const canManage = result?.your_permission === 'admin'

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{r.cloudMembersTitle}</DialogTitle>
          <DialogDescription>{r.cloudMembersDesc}</DialogDescription>
        </DialogHeader>
        <div className="max-h-72 space-y-2 overflow-y-auto">
          {loading ? (
            <div className="text-sm text-muted-foreground">{t.common.loading}</div>
          ) : members.length === 0 ? (
            <div className="text-sm text-muted-foreground">{r.cloudMembersEmpty}</div>
          ) : (
            members.map(member => {
              const permission = cloudMemberPermission(member.permission)
              const busy =
                pendingMemberId === `permission:${member.account_id}` ||
                pendingMemberId === `remove:${member.account_id}`

              return (
                <div className="rounded-md border border-border/70 px-3 py-2" key={member.account_id}>
                  <div className="truncate text-sm font-medium">
                    {member.display_name || member.email || member.account_id}
                  </div>
                  <div className="mt-0.5 flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
                    {member.email ? <span className="min-w-0 truncate">{member.email}</span> : null}
                    <span className="shrink-0">{member.granted_via || 'invite'}</span>
                  </div>
                  <div className="mt-2 flex items-center gap-2">
                    <Select
                      disabled={!canManage || busy}
                      onValueChange={value => void updatePermission(member, value as CloudChannelPermission)}
                      value={permission}
                    >
                      <SelectTrigger
                        aria-label={r.cloudMembersPermissionLabel}
                        className="h-7 w-28"
                        size="sm"
                      >
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {CLOUD_MEMBER_PERMISSIONS.map(option => (
                          <SelectItem key={option} value={option}>
                            {r.cloudMembersPermission(option)}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Tip label={r.cloudMembersRevoke}>
                      <Button
                        aria-label={r.cloudMembersRevoke}
                        disabled={!canManage || busy}
                        onClick={() => void removeMember(member)}
                        size="icon-xs"
                        type="button"
                        variant="ghost"
                      >
                        <Codicon name="trash" />
                      </Button>
                    </Tip>
                    {busy ? <span className="text-xs text-muted-foreground">{r.cloudMembersSaving}</span> : null}
                  </div>
                </div>
              )
            })
          )}
        </div>
        <DialogFooter>
          <Button disabled={loading} onClick={() => onOpenChange(false)} type="button" variant="ghost">
            {t.common.close}
          </Button>
          <Button disabled={loading} onClick={() => void refreshMembers()} type="button">
            {r.cloudMembersRefresh}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
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
