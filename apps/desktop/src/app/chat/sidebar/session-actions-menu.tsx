import { IconBookmark, IconBookmarkFilled, IconCircleX, IconFileDownload, IconPencil } from '@tabler/icons-react'
import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'
import type * as React from 'react'
import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/ui/copy-button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { renameSession } from '@/hermes'
import { triggerHaptic } from '@/lib/haptics'
import { exportSession } from '@/lib/session-export'
import { cn } from '@/lib/utils'
import { $desktopLanguage } from '@/store/language'
import { notify, notifyError } from '@/store/notifications'
import { setSessions } from '@/store/session'

interface SessionActionsMenuProps extends Pick<
  React.ComponentProps<typeof DropdownMenuContent>,
  'align' | 'sideOffset'
> {
  children: ReactNode
  title: string
  sessionId: string
  pinned?: boolean
  onPin?: () => void
  onDelete?: () => void
}

export function SessionActionsMenu({
  children,
  title,
  sessionId,
  pinned = false,
  onPin,
  onDelete,
  align = 'end',
  sideOffset = 6
}: SessionActionsMenuProps) {
  const language = useStore($desktopLanguage)
  const itemClass = 'gap-2.5 text-foreground focus:bg-accent [&_svg]:size-4'
  const [renameOpen, setRenameOpen] = useState(false)

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>{children}</DropdownMenuTrigger>
        <DropdownMenuContent
          align={align}
          aria-label={language === 'zh' ? `${title} 的操作` : `Actions for ${title}`}
          className="w-44"
          sideOffset={sideOffset}
        >
          <DropdownMenuItem
            className={itemClass}
            disabled={!onPin}
            onSelect={() => {
              triggerHaptic('selection')
              onPin?.()
            }}
          >
            {pinned ? <IconBookmarkFilled /> : <IconBookmark />}
            <span>{language === 'zh' ? (pinned ? '取消置顶' : '置顶') : pinned ? 'Unpin' : 'Pin'}</span>
          </DropdownMenuItem>
          <CopyButton
            appearance="menu-item"
            className={itemClass}
            disabled={!sessionId}
            errorMessage={language === 'zh' ? '无法复制会话 ID' : 'Could not copy session ID'}
            label={language === 'zh' ? '复制 ID' : 'Copy ID'}
            text={sessionId}
          />
          <DropdownMenuItem
            className={itemClass}
            disabled={!sessionId}
            onSelect={() => {
              triggerHaptic('selection')
              void exportSession(sessionId, { title })
            }}
          >
            <IconFileDownload />
            <span>{language === 'zh' ? '导出' : 'Export'}</span>
          </DropdownMenuItem>
          <DropdownMenuItem
            className={itemClass}
            disabled={!sessionId}
            onSelect={() => {
              triggerHaptic('selection')
              setRenameOpen(true)
            }}
          >
            <IconPencil />
            <span>{language === 'zh' ? '重命名' : 'Rename'}</span>
          </DropdownMenuItem>
          <DropdownMenuSeparator className="my-3" />
          <DropdownMenuItem
            className={cn(itemClass, 'text-destructive focus:text-destructive')}
            disabled={!onDelete}
            onSelect={() => {
              triggerHaptic('warning')
              onDelete?.()
            }}
            variant="destructive"
          >
            <IconCircleX />
            <span>{language === 'zh' ? '删除' : 'Delete'}</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <RenameSessionDialog
        currentTitle={title}
        language={language}
        onOpenChange={setRenameOpen}
        open={renameOpen}
        sessionId={sessionId}
      />
    </>
  )
}

interface RenameSessionDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  sessionId: string
  currentTitle: string
  language: 'zh' | 'en'
}

function RenameSessionDialog({ open, onOpenChange, sessionId, currentTitle, language }: RenameSessionDialogProps) {
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
      notify({ kind: 'success', message: language === 'zh' ? '已重命名' : 'Renamed', durationMs: 2_000 })
      onOpenChange(false)
    } catch (err) {
      notifyError(err, language === 'zh' ? '重命名失败' : 'Rename failed')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{language === 'zh' ? '重命名会话' : 'Rename session'}</DialogTitle>
          <DialogDescription>
            {language === 'zh' ? '给这个对话设置一个便于识别的标题。留空可清除标题。' : 'Give this chat a memorable title. Leave empty to clear.'}
          </DialogDescription>
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
          placeholder={language === 'zh' ? '未命名会话' : 'Untitled session'}
          ref={inputRef}
          value={value}
        />
        <DialogFooter>
          <Button disabled={submitting} onClick={() => onOpenChange(false)} type="button" variant="ghost">
            {language === 'zh' ? '取消' : 'Cancel'}
          </Button>
          <Button disabled={submitting} onClick={() => void submit()} type="button">
            {language === 'zh' ? '保存' : 'Save'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
