import { useStore } from '@nanostores/react'
import type * as React from 'react'

import { writeSessionDrag } from '@/app/chat/composer/inline-refs'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import type { SessionInfo } from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'
import { $attentionSessionIds, sessionPinId } from '@/store/session'
import type { SessionPresenceRecord } from '@/types/hermes'

import { SessionActionsMenu, SessionContextMenu } from './session-actions-menu'

export interface SidebarSessionRowProps extends React.ComponentProps<'div'> {
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
  /** Presence record for this session (from another device) — indicates live/active state. */
  presence?: SessionPresenceRecord
}

const AGE_TICKS: ReadonlyArray<[number, 'ageDay' | 'ageHour' | 'ageMin']> = [
  [86_400_000, 'ageDay'],
  [3_600_000, 'ageHour'],
  [60_000, 'ageMin']
]

function formatAge(seconds: number, r: Translations['sidebar']['row']): string {
  const delta = Math.max(0, Date.now() - seconds * 1000)

  for (const [ms, key] of AGE_TICKS) {
    if (delta >= ms) {
      return `${Math.floor(delta / ms)}${r[key]}`
    }
  }

  return r.ageNow
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
  presence,
  reorderable = false,
  dragging = false,
  dragHandleProps,
  className,
  style,
  ref,
  ...rest
}: SidebarSessionRowProps) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const title = sessionTitle(session)
  const age = formatAge(session.last_active || session.started_at, r)
  const handleLabel = `Reorder ${title}`
  // Subscribe per-row (the leaf) instead of drilling a set through the list —
  // the atom is tiny and rarely non-empty. True when a clarify prompt in this
  // session is waiting on the user.
  const needsInput = useStore($attentionSessionIds).includes(session.id)

  return (
    <SessionContextMenu
      onArchive={onArchive}
      onDelete={onDelete}
      onPin={onPin}
      pinned={isPinned}
      profile={session.profile}
      sessionId={session.id}
      title={title}
    >
      <div
        className={cn(
          'group relative grid min-h-[1.625rem] cursor-pointer grid-cols-[minmax(0,1fr)_auto] items-center rounded-md transition-colors duration-100 ease-out hover:bg-(--ui-row-hover-background) hover:transition-none',
          isSelected && 'bg-(--ui-row-active-background)',
          isWorking && 'text-foreground',
          dragging && 'z-10 cursor-grabbing opacity-60 shadow-sm',
          className
        )}
        data-session-id={session.id}
        data-working={isWorking ? 'true' : undefined}
        draggable
        onDragStart={event => {
          // Reorder drags belong to dnd-kit (the grab handle) — cancel the
          // native drag so the two DnD systems don't fight.
          if ((event.target as HTMLElement).closest('[data-reorder-handle]')) {
            event.preventDefault()

            return
          }

          writeSessionDrag(event.dataTransfer, {
            id: session.id,
            pinId: sessionPinId(session),
            pinned: isPinned,
            profile: session.profile || 'default',
            title
          })
        }}
        ref={ref}
        style={style}
        {...rest}
      >
        {isWorking && !needsInput && <span aria-hidden="true" className="arc-border" />}
        <button
          className="z-0 flex min-w-0 items-center gap-1.5 bg-transparent py-0.5 pl-2 pr-2 text-left"
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
              className={cn(
                // Scope the dot↔grabber swap to a local group so the grabber
                // only reveals when hovering/focusing the handle itself, not
                // anywhere on the row. Width MUST match the non-reorderable dot
                // column (w-3.5) so rows don't shift horizontally when reorder is
                // toggled (e.g. scoped → ALL-profiles view).
                'group/handle relative -my-0.5 grid w-3.5 shrink-0 cursor-grab touch-none place-items-center self-stretch overflow-hidden active:cursor-grabbing',
                // The quest-glow box-shadow extends past the dot; let it bleed
                // out instead of being clipped by this handle's overflow-hidden.
                needsInput && 'overflow-visible'
              )}
              data-reorder-handle
              onClick={event => event.stopPropagation()}
            >
              <SidebarRowDot
                className="transition-opacity group-hover/handle:opacity-0 group-focus-within/handle:opacity-0"
                isWorking={isWorking}
                needsInput={needsInput}
              />
              <Codicon
                className={cn(
                  'absolute text-(--ui-text-quaternary) opacity-0 transition-opacity group-hover/handle:opacity-80 group-focus-within/handle:opacity-80 hover:text-(--ui-text-secondary)',
                  dragging && 'text-(--ui-text-secondary) opacity-100'
                )}
                name="grabber"
                size="0.75rem"
              />
            </span>
          ) : (
            <span
              className={cn(
                'grid w-3.5 shrink-0 place-items-center',
                needsInput ? 'overflow-visible' : 'overflow-hidden',
                'self-center'
              )}
            >
            <SidebarRowDot isWorking={isWorking} needsInput={needsInput} />
          </span>
          )}
          <div className="min-w-0 flex-1">
            <span className="block truncate text-[0.8125rem] font-normal text-(--ui-text-secondary) group-hover:text-foreground group-data-[working=true]:text-foreground/90">{title}</span>
          </div>
        </button>
        {/* Trailing slot: on an IDLE row the timestamp is visible and, on
            hover/focus, slides left to make room while the 3-dot menu slides
            in from the right (both on screen at once). On an ACTIVE row the
            pulsing orange dot on the left already signals "running", so the
            timestamp is hidden — but its width is still reserved (opacity-0,
            not unmounted) so the menu lands in the same spot and the row
            height never shifts. Transform/opacity only — no layout reflow. */}
        <div className="relative flex h-full items-center justify-end self-stretch pl-1 pr-1.5">
          <span
            className={cn(
              'pointer-events-none min-w-6 text-right text-[0.625rem] leading-none text-(--ui-text-tertiary) transition-[transform,opacity] duration-150 ease-out',
              // Slide left by the menu's footprint on hover so the age stays
              // fully legible beside the revealed 3-dot button.
              'group-hover:-translate-x-5 group-focus-within:-translate-x-5',
              // Active sessions: the orange dot is the status cue; hide the
              // timestamp (keep its reserved width) for the whole active run.
              isWorking && 'opacity-0'
            )}
          >
            {age}
          </span>
          <div className="absolute inset-y-0 right-1 grid place-items-center">
            <SessionActionsMenu
              onArchive={onArchive}
              onDelete={onDelete}
              onPin={onPin}
              pinned={isPinned}
              profile={session.profile}
              sessionId={session.id}
              title={title}
            >
              <Button
                aria-label={r.actionsFor(title)}
                className="size-5 translate-x-1 scale-90 rounded-[4px] bg-transparent text-transparent opacity-0 transition-all duration-150 ease-out group-hover:translate-x-0 group-hover:scale-100 group-hover:text-(--ui-text-tertiary) group-hover:opacity-100 group-focus-within:translate-x-0 group-focus-within:scale-100 group-focus-within:opacity-100 hover:bg-(--ui-control-active-background)! hover:text-foreground! focus-visible:bg-(--ui-control-active-background) focus-visible:text-foreground focus-visible:opacity-100 focus-visible:ring-0 data-[state=open]:translate-x-0 data-[state=open]:scale-100 data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground data-[state=open]:opacity-100 [&_svg]:size-3.5!"
                size="icon"
                title={r.sessionActions}
                variant="ghost"
              >
                <Codicon name="ellipsis" size="0.875rem" />
              </Button>
            </SessionActionsMenu>
          </div>
        </div>
      </div>
    </SessionContextMenu>
  )
}

function SidebarRowDot({
  isWorking,
  needsInput = false,
  className
}: {
  isWorking: boolean
  needsInput?: boolean
  className?: string
}) {
  const { t } = useI18n()
  const r = t.sidebar.row

  // "Needs input" wins over "working": a clarify-blocked session is technically
  // still running, but the actionable state is that it's waiting on the user.
  // Amber + steady (no ping) reads as "your turn", distinct from the accent
  // pulse of an active turn.
  if (needsInput) {
    return (
      <span
        aria-label={r.needsInput}
        className={cn('quest-glow relative size-1.5 rounded-full bg-amber-500', className)}
        role="status"
        title={r.waitingForAnswer}
      />
    )
  }

  return (
    <span
      aria-label={isWorking ? r.sessionRunning : undefined}
      className={cn(
        'rounded-full',
        isWorking
          ? "relative size-1.5 bg-orange-500 shadow-[0_0_0.625rem_color-mix(in_srgb,#f97316_60%,transparent)] before:absolute before:inset-0 before:animate-ping before:rounded-full before:bg-orange-500 before:opacity-70 before:content-['']"
          : 'size-1 bg-(--ui-text-quaternary) opacity-80',
        className
      )}
      role={isWorking ? 'status' : undefined}
    />
  )
}
