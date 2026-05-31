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
import { triggerHaptic } from '@/lib/haptics'
import { exportSession } from '@/lib/session-export'
import { notify, notifyError } from '@/store/notifications'
import { setSessions } from '@/store/session'

interface SessionActions {
  sessionId: string
  title: string
  pinned?: boolean
  onPin?: () => void
  onDelete?: () => void
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

function useSessionActions({ sessionId, title, pinned = false, onPin, onDelete }: SessionActions) {
  const [renameOpen, setRenameOpen] = useState(false)

  const items: ItemSpec[] = [
    {
      disabled: !onPin,
      icon: 'pin',
      label: pinned ? 'Unpin' : 'Pin',
      onSelect: () => {
        triggerHaptic('selection')
        onPin?.()
      }
    },
    {
      disabled: !sessionId,
      icon: 'copy',
      label: 'Copy ID',
      onSelect: event => {
        event.preventDefault()
        triggerHaptic('selection')
        void writeClipboardText(sessionId).catch(err => notifyError(err, 'Could not copy session ID'))
      }
    },
    {
      disabled: !sessionId,
      icon: 'cloud-download',
      label: 'Export',
      onSelect: () => {
        triggerHaptic('selection')
        void exportSession(sessionId, { title })
      }
    },
    {
      disabled: !sessionId,
      icon: 'edit',
      label: 'Rename',
      onSelect: () => {
        triggerHaptic('selection')
        setRenameOpen(true)
      }
    },
    {
      className: 'text-destructive focus:text-destructive',
      disabled: !onDelete,
      icon: 'trash',
      label: 'Delete',
      onSelect: () => {
        triggerHaptic('warning')
        onDelete?.()
      },
      variant: 'destructive'
    }
  ]

  const renderItems = (Item: MenuItem) =>
    items.map(({ className, disabled, icon, label, onSelect, variant }) => (
      <Item className={className} disabled={disabled} key={label} onSelect={onSelect} variant={variant}>
        <Codicon name={icon} size="0.875rem" />
        <span>{label}</span>
      </Item>
    ))

  const renameDialog = (
    <RenameSessionDialog currentTitle={title} onOpenChange={setRenameOpen} open={renameOpen} sessionId={sessionId} />
  )

  return { renameDialog, renderItems }
}

interface SessionActionsMenuProps
  extends SessionActions, Pick<React.ComponentProps<typeof DropdownMenuContent>, 'align' | 'sideOffset'> {
  children: React.ReactNode
}

export function SessionActionsMenu({ children, align = 'end', sideOffset = 6, ...actions }: SessionActionsMenuProps) {
  const { renameDialog, renderItems } = useSessionActions(actions)

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>{children}</DropdownMenuTrigger>
        <DropdownMenuContent
          align={align}
          aria-label={`Actions for ${actions.title}`}
          className="w-40"
          sideOffset={sideOffset}
        >
          {renderItems(DropdownMenuItem)}
        </DropdownMenuContent>
      </DropdownMenu>
      {renameDialog}
    </>
  )
}

interface SessionContextMenuProps extends SessionActions {
  children: React.ReactNode
}

export function SessionContextMenu({ children, ...actions }: SessionContextMenuProps) {
  const { renameDialog, renderItems } = useSessionActions(actions)

  return (
    <>
      <ContextMenu>
        <ContextMenuTrigger asChild>{children}</ContextMenuTrigger>
        <ContextMenuContent aria-label={`Actions for ${actions.title}`} className="w-40">
          {renderItems(ContextMenuItem)}
        </ContextMenuContent>
      </ContextMenu>
      {renameDialog}
    </>
  )
}

interface RenameSessionDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  sessionId: string
  currentTitle: string
}

function RenameSessionDialog({ open, onOpenChange, sessionId, currentTitle }: RenameSessionDialogProps) {
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
      const result = await renameSession(sessionId, next)
      const finalTitle = result.title || next || ''
      setSessions(prev => prev.map(s => (s.id === sessionId ? { ...s, title: finalTitle || null } : s)))
      notify({ durationMs: 2_000, kind: 'success', message: 'Renamed' })
      onOpenChange(false)
    } catch (err) {
      notifyError(err, 'Rename failed')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Rename session</DialogTitle>
          <DialogDescription>Give this chat a memorable title. Leave empty to clear.</DialogDescription>
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
          placeholder="Untitled session"
          ref={inputRef}
          value={value}
        />
        <DialogFooter>
          <Button disabled={submitting} onClick={() => onOpenChange(false)} type="button" variant="ghost">
            Cancel
          </Button>
          <Button disabled={submitting} onClick={() => void submit()} type="button">
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
