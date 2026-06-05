import type * as React from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
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
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>{children}</DropdownMenuTrigger>
      <DropdownMenuContent
        align={align}
        aria-label={`Actions for ${title}`}
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
          <span>{isPaused ? 'Resume' : 'Pause'}</span>
        </DropdownMenuItem>

        <DropdownMenuItem
          disabled={busy}
          onSelect={() => {
            triggerHaptic('selection')
            onTrigger()
          }}
        >
          <Codicon name="zap" size="0.875rem" />
          <span>Trigger now</span>
        </DropdownMenuItem>

        <DropdownMenuItem
          onSelect={() => {
            triggerHaptic('selection')
            onEdit()
          }}
        >
          <Codicon name="edit" size="0.875rem" />
          <span>Edit</span>
        </DropdownMenuItem>

        <DropdownMenuItem
          onSelect={() => {
            triggerHaptic('warning')
            onDelete()
          }}
          variant="destructive"
        >
          <Codicon name="trash" size="0.875rem" />
          <span>Delete</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

interface CronJobActionsTriggerProps extends Omit<React.ComponentProps<typeof Button>, 'size' | 'variant'> {
  title: string
}

export function CronJobActionsTrigger({ className, title, ...props }: CronJobActionsTriggerProps) {
  return (
    <Button
      aria-label={`Actions for ${title}`}
      className={className}
      size="icon-sm"
      title="Cron job actions"
      variant="ghost"
      {...props}
    >
      <Codicon className="text-muted-foreground" name="ellipsis" size="0.875rem" />
    </Button>
  )
}
