import type * as React from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'

interface CronJobActions {
  busy?: boolean
  isPaused: boolean
  title: string
  onDelete: () => void
  onEdit: () => void
  onPauseResume: () => void
  onTrigger: () => void
}

interface CronJobActionsMenuProps
  extends CronJobActions, Pick<React.ComponentProps<typeof DropdownMenuContent>, 'align' | 'sideOffset'> {
  children: React.ReactNode
}

export function CronJobActionsMenu({
  align = 'end',
  busy = false,
  children,
  isPaused,
  onDelete,
  onEdit,
  onPauseResume,
  onTrigger,
  sideOffset = 6,
  title
}: CronJobActionsMenuProps) {
  const { t } = useI18n()
  const c = t.cron

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>{children}</DropdownMenuTrigger>
      <DropdownMenuContent
        align={align}
        aria-label={c.actionsFor(title)}
        className="w-44"
        sideOffset={sideOffset}
      >
        <DropdownMenuItem
          disabled={busy}
          onSelect={() => {
            triggerHaptic('selection')
            onPauseResume()
          }}
        >
          <Codicon name={isPaused ? 'play' : 'debug-pause'} size="0.875rem" />
          <span>{isPaused ? c.resumeTitle : c.pauseTitle}</span>
        </DropdownMenuItem>

        <DropdownMenuItem
          disabled={busy}
          onSelect={() => {
            triggerHaptic('selection')
            onTrigger()
          }}
        >
          <Codicon name="zap" size="0.875rem" />
          <span>{c.triggerNow}</span>
        </DropdownMenuItem>

        <DropdownMenuItem
          onSelect={() => {
            triggerHaptic('selection')
            onEdit()
          }}
        >
          <Codicon name="edit" size="0.875rem" />
          <span>{c.edit}</span>
        </DropdownMenuItem>

        <DropdownMenuItem
          onSelect={() => {
            triggerHaptic('warning')
            onDelete()
          }}
          variant="destructive"
        >
          <Codicon name="trash" size="0.875rem" />
          <span>{t.common.delete}</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

interface CronJobActionsTriggerProps extends Omit<React.ComponentProps<typeof Button>, 'size' | 'variant'> {
  title: string
}

export function CronJobActionsTrigger({ className, title, ...props }: CronJobActionsTriggerProps) {
  const { t } = useI18n()

  return (
    <Button
      aria-label={t.cron.actionsFor(title)}
      className={className}
      size="icon-sm"
      title={t.cron.actionsTitle}
      variant="ghost"
      {...props}
    >
      <Codicon className="text-muted-foreground" name="ellipsis" size="0.875rem" />
    </Button>
  )
}
