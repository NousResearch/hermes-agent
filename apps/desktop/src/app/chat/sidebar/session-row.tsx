import type * as React from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import type { SessionInfo } from '@/hermes'
import { sessionTitle } from '@/lib/chat-runtime'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'

import { SessionActionsMenu, SessionContextMenu } from './session-actions-menu'

interface SidebarSessionRowProps extends React.ComponentProps<'div'> {
  session: SessionInfo
  isPinned: boolean
  isSelected: boolean
  isWorking: boolean
  onArchive: () => void
  onDelete: () => void
  onPin: () => void
  onResume: () => void
  reorderable?: boolean
  dragging?: boolean
  dragHandleProps?: React.HTMLAttributes<HTMLElement>
}

const AGE_TICKS: ReadonlyArray<[number, string]> = [
  [86_400_000, 'd'],
  [3_600_000, 'h'],
  [60_000, 'm']
]

function formatAge(seconds: number): string {
  const delta = Math.max(0, Date.now() - seconds * 1000)

  for (const [ms, suffix] of AGE_TICKS) {
    if (delta >= ms) {
      return `${Math.floor(delta / ms)}${suffix}`
    }
  }

  return 'now'
}

export function SidebarSessionRow({
  session,
  isPinned,
  isSelected,
  isWorking,
  onArchive,
  onDelete,
  onPin,
  onResume,
  reorderable = false,
  dragging = false,
  dragHandleProps,
  className,
  style,
  ref,
  ...rest
}: SidebarSessionRowProps) {
  const title = sessionTitle(session)
  const age = formatAge(session.last_active || session.started_at)
  const handleLabel = `Reorder ${title}`

  return (
    <SessionContextMenu
      onArchive={onArchive}
      onDelete={onDelete}
      onPin={onPin}
      pinned={isPinned}
      sessionId={session.id}
      title={title}
    >
      <div
        className={cn(
          'group relative grid min-h-[1.625rem] cursor-pointer grid-cols-[minmax(0,1fr)_1.375rem] items-center rounded-md transition-colors duration-100 ease-out hover:bg-(--ui-row-hover-background) hover:transition-none',
          isSelected && 'bg-(--ui-row-active-background)',
          isWorking && 'text-foreground',
          dragging && 'z-10 cursor-grabbing opacity-60 shadow-sm',
          className
        )}
        data-working={isWorking ? 'true' : undefined}
        ref={ref}
        style={style}
        {...rest}
      >
        {isWorking && <span aria-hidden="true" className="arc-border" />}
        <button
          className="z-0 flex min-w-0 cursor-pointer items-center gap-1.5 bg-transparent py-0.5 pl-2 pr-1 text-left group-hover:pr-12"
          onClick={event => {
            if (event.shiftKey) {
              event.preventDefault()
              event.stopPropagation()
              triggerHaptic('selection')
              onPin()

              return
            }

            if (event.metaKey || event.ctrlKey) {
              event.preventDefault()
              event.stopPropagation()
              triggerHaptic('selection')
              onArchive()

              return
            }

            onResume()
          }}
          type="button"
        >
          {reorderable ? (
            <span
              {...dragHandleProps}
              aria-label={handleLabel}
              className="relative -my-0.5 grid w-4 shrink-0 cursor-grab touch-none place-items-center self-stretch overflow-hidden active:cursor-grabbing"
              onClick={event => event.stopPropagation()}
            >
              <SidebarRowDot
                className="transition-opacity group-hover:opacity-0 group-focus-within:opacity-0"
                isWorking={isWorking}
              />
              <Codicon
                className={cn(
                  'absolute text-(--ui-text-quaternary) opacity-0 transition-opacity group-hover:opacity-80 group-focus-within:opacity-80 hover:text-(--ui-text-secondary)',
                  dragging && 'text-(--ui-text-secondary) opacity-100'
                )}
                name="grabber"
                size="0.75rem"
              />
            </span>
          ) : (
            <span className="grid w-3.5 shrink-0 place-items-center overflow-hidden">
              <SidebarRowDot isWorking={isWorking} />
            </span>
          )}
          <span className="truncate text-[0.8125rem] font-normal text-(--ui-text-secondary) group-hover:text-foreground group-data-[working=true]:text-foreground/90">
            {title}
          </span>
        </button>
        <div className="relative z-2 grid w-[1.375rem] place-items-center">
          {!isWorking && (
            <span className="pointer-events-none absolute right-6 top-1/2 min-w-6 -translate-y-1/2 text-right text-[0.625rem] leading-none text-(--ui-text-tertiary) opacity-0 transition-opacity group-hover:opacity-100">
              {age}
            </span>
          )}
          <SessionActionsMenu
            onArchive={onArchive}
            onDelete={onDelete}
            onPin={onPin}
            pinned={isPinned}
            sessionId={session.id}
            title={title}
          >
            <Button
              aria-label={`Actions for ${title}`}
              className="size-5 rounded-md bg-transparent text-transparent transition-colors duration-100 hover:bg-(--ui-control-active-background) hover:text-foreground focus-visible:bg-(--ui-control-active-background) focus-visible:text-foreground focus-visible:ring-0 data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground group-hover:text-(--ui-text-tertiary) [&_svg]:size-3.5!"
              size="icon"
              title="Session actions"
              variant="ghost"
            >
              <Codicon name="ellipsis" size="0.875rem" />
            </Button>
          </SessionActionsMenu>
        </div>
      </div>
    </SessionContextMenu>
  )
}

function SidebarRowDot({ isWorking, className }: { isWorking: boolean; className?: string }) {
  return (
    <span
      aria-label={isWorking ? 'Session running' : undefined}
      className={cn(
        'rounded-full',
        isWorking
          ? "relative size-1.5 bg-(--ui-accent) shadow-[0_0_0.625rem_color-mix(in_srgb,var(--ui-accent)_55%,transparent)] before:absolute before:inset-0 before:animate-ping before:rounded-full before:bg-(--ui-accent) before:opacity-70 before:content-['']"
          : 'size-1 bg-(--ui-text-quaternary) opacity-80',
        className
      )}
      role={isWorking ? 'status' : undefined}
    />
  )
}
