import { useStore } from '@nanostores/react'
import type * as React from 'react'

import { writeSessionDrag } from '@/app/chat/composer/inline-refs'
import { PlatformAvatar } from '@/app/messaging/platform-icon'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { triggerHaptic } from '@/lib/haptics'
import { handoffOriginSource, sessionSourceLabel } from '@/lib/session-source'
import { cn } from '@/lib/utils'
import { $attentionSessionIds, $unreadSessionIds, isSessionUnread } from '@/store/session'
import { canOpenSessionWindow, openSessionInNewWindow } from '@/store/windows'

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
  // A handed-off session's live source is local, but it originated on a
  // messaging platform — surface that origin as a small badge so e.g. a
  // Telegram thread continued here still reads as Telegram.
  const handoffSource = handoffOriginSource(session.handoff_state, session.handoff_platform)
  const handoffLabel = handoffSource ? sessionSourceLabel(handoffSource) ?? handoffSource : null
  // Subscribe per-row (the leaf) instead of drilling a set through the list —
  // the atom is tiny and rarely non-empty. True when a clarify prompt in this
  // session is waiting on the user.
  const needsInput = useStore($attentionSessionIds).includes(session.id)
  // "Unread" = the session produced output (turn done / errored / needs input)
  // that the user hasn't focused since. Subscribing to $unreadSessionIds
  // re-renders on any change; isSessionUnread then resolves the lineage root
  // so a compression rotation can't hide the marker.
  useStore($unreadSessionIds)
  const unread = isSessionUnread(session.id)

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
          'group relative grid min-h-[1.625rem] cursor-pointer grid-cols-[minmax(0,1fr)_1.375rem] items-center rounded-md transition-colors duration-100 ease-out hover:bg-(--ui-row-hover-background) hover:transition-none',
          isSelected && 'bg-(--ui-row-active-background)',
          (isWorking || unread) && 'text-foreground',
          // Opaque surface while lifted so the dragged row erases what's under
          // it (translucency let the rows below bleed through).
          dragging && 'z-10 cursor-grabbing bg-(--ui-sidebar-surface-background)',
          className
        )}
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
          className="z-0 flex min-w-0 items-center gap-1.5 bg-transparent py-0.5 pl-2 pr-1 text-left group-hover:pr-12"
          onClick={event => {
            if (event.shiftKey) {
              event.preventDefault()
              event.stopPropagation()
              triggerHaptic('selection')
              onPin()

              return
            }

            // ⌘-click (mac) / ⌃-click (win/linux) pops the chat into its own
            // window — the universal "open in a new window" gesture. Archive
            // lives in the row's ⋯ and right-click menus. Falls through to a
            // normal resume when standalone windows aren't available (web embed).
            if ((event.metaKey || event.ctrlKey) && canOpenSessionWindow()) {
              event.preventDefault()
              event.stopPropagation()
              triggerHaptic('selection')
              void openSessionInNewWindow(session.id)

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
                unread={unread}
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
                needsInput ? 'overflow-visible' : 'overflow-hidden'
              )}
            >
              <SidebarRowDot isWorking={isWorking} needsInput={needsInput} unread={unread} />
            </span>
          )}
          {handoffSource && handoffLabel ? (
            <Tip label={r.handoffOrigin(handoffLabel)}>
              <PlatformAvatar
                className="size-4 rounded-[4px] text-[0.5rem] [&_svg]:size-2.5"
                platformId={handoffSource}
                platformName={handoffLabel}
              />
            </Tip>
          ) : null}
          <span
            className={cn(
              'min-w-0 flex-1 truncate text-[0.8125rem] font-normal text-(--ui-text-secondary) group-hover:text-foreground group-data-[working=true]:text-foreground/90',
              // Unread rows brighten to full foreground + semibold — the
              // "unread email" weight cue, distinct from the accent pulse of
              // a working turn (same color, motion is the differentiator).
              unread && 'font-medium text-foreground'
            )}
          >
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
            profile={session.profile}
            sessionId={session.id}
            title={title}
          >
            <Button
              aria-label={r.actionsFor(title)}
              className="size-5 rounded-[4px] bg-transparent text-transparent transition-colors duration-100 hover:bg-(--ui-control-active-background) hover:text-foreground focus-visible:bg-(--ui-control-active-background) focus-visible:text-foreground focus-visible:ring-0 data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground group-hover:text-(--ui-text-tertiary) [&_svg]:size-3.5!"
              size="icon"
              title={r.sessionActions}
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

function SidebarRowDot({
  isWorking,
  needsInput = false,
  unread = false,
  className
}: {
  isWorking: boolean
  needsInput?: boolean
  unread?: boolean
  className?: string
}) {
  const { t } = useI18n()
  const r = t.sidebar.row

  // Priority ladder: needsInput > unread > working > idle.
  //  - needsInput: amber + steady quest-glow ("your turn") — highest urgency.
  //  - unread: solid accent, steady (no pulse) — "new output waiting".
  //    Same color as working; the distinction is motion (steady vs. pulse),
  //    not hue, so it doesn't add a new color to the restrained palette.
  //  - working: accent + ping pulse ("thinking").
  //  - idle: dim gray.
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

  if (unread) {
    return (
      <span
        aria-label={r.unread}
        className={cn('relative size-1.5 rounded-full bg-(--ui-accent)', className)}
        role="status"
        title={r.unread}
      />
    )
  }

  return (
    <span
      aria-label={isWorking ? r.sessionRunning : undefined}
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
