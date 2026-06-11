import { useStore } from '@nanostores/react'
import { motion } from 'motion/react'
import { useState } from 'react'
import type * as React from 'react'

import { type SessionDragPayload, writeSessionDrag } from '@/app/chat/composer/inline-refs'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import type { SessionInfo } from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'
import { $attentionSessionIds, sessionPinId } from '@/store/session'
import { canOpenSessionWindow, openSessionInNewWindow } from '@/store/windows'
import type { SessionPresenceRecord } from '@/types/hermes'

import { SessionActionsMenu, type SessionBulkContextActions, SessionContextMenu } from './session-actions-menu'

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
  onSessionDragEnd?: () => void
  onSessionDragStart?: (payload: SessionDragPayload) => void
  /** Presence record for this session (from another device) — indicates live/active state. */
  presence?: SessionPresenceRecord
  /** Row renders in the Archived section: menus swap Archive→Restore, pin
   * gestures are disabled (a pin can't resolve an archived row), and the drag
   * payload carries the archived marker so drop zones can offer "restore". */
  archived?: boolean
  onRestore?: () => void
  /** Row participates in its section's multi-select. */
  selectable?: boolean
  /** ≥1 row in THIS section is selected → clicks toggle membership instead of
   * resuming, and the leading slot becomes a checkbox. */
  selectionActive?: boolean
  /** This row is in the current selection. */
  checked?: boolean
  onToggleSelect?: (mode: 'range' | 'single') => void
  /** Multi-selected ids to use for the row's right-click menu when this row is checked. */
  bulkSelectedSessionIds?: readonly string[]
  onArchiveSelectedSessions?: SessionBulkContextActions['onArchiveSessions']
  onDeleteSelectedSessions?: SessionBulkContextActions['onDeleteSessions']
  onHaltSelectedSessions?: SessionBulkContextActions['onHaltSessions']
  onPromptSelectedSessions?: SessionBulkContextActions['onPromptSessions']
  onRestoreSelectedSessions?: SessionBulkContextActions['onRestoreSessions']
  onSteerSelectedSessions?: SessionBulkContextActions['onSteerSessions']
}

const AGE_TICKS: ReadonlyArray<[number, 'ageDay' | 'ageHour' | 'ageMin']> = [
  [86_400_000, 'ageDay'],
  [3_600_000, 'ageHour'],
  [60_000, 'ageMin']
]

export function formatSidebarRowAge(seconds: number, r: Translations['sidebar']['row']): string {
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
  onSessionDragEnd,
  onSessionDragStart,
  archived = false,
  onRestore,
  selectable = false,
  selectionActive = false,
  checked = false,
  onToggleSelect,
  bulkSelectedSessionIds,
  onArchiveSelectedSessions,
  onDeleteSelectedSessions,
  onHaltSelectedSessions,
  onPromptSelectedSessions,
  onRestoreSelectedSessions,
  onSteerSelectedSessions,
  className,
  style,
  ref,
  ...rest
}: SidebarSessionRowProps) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const title = sessionTitle(session)
  const age = formatSidebarRowAge(session.last_active || session.started_at, r)
  const [actionsMenuOpen, setActionsMenuOpen] = useState(false)
  const [actionsKeyboardFocus, setActionsKeyboardFocus] = useState(false)
  const actionsVisible = actionsMenuOpen || actionsKeyboardFocus

  const bulkContextActions =
    checked && bulkSelectedSessionIds && bulkSelectedSessionIds.length > 1
      ? {
          archived,
          onArchiveSessions: onArchiveSelectedSessions,
          onDeleteSessions: onDeleteSelectedSessions,
          onHaltSessions: onHaltSelectedSessions,
          onPromptSessions: onPromptSelectedSessions,
          onRestoreSessions: onRestoreSelectedSessions,
          onSteerSessions: onSteerSelectedSessions,
          sessionIds: bulkSelectedSessionIds
        }
      : undefined

  // Subscribe per-row (the leaf) instead of drilling a set through the list —
  // the atom is tiny and rarely non-empty. True when a clarify prompt in this
  // session is waiting on the user.
  const needsInput = useStore($attentionSessionIds).includes(session.id)

  const toggleSelect = (mode: 'range' | 'single') => {
    triggerHaptic('selection')
    onToggleSelect?.(mode)
  }

  return (
    <SessionContextMenu
      archived={archived}
      bulkActions={bulkContextActions}
      onArchive={onArchive}
      onDelete={onDelete}
      onPin={onPin}
      onRestore={onRestore}
      onSelect={selectable && onToggleSelect ? () => toggleSelect('single') : undefined}
      pinned={isPinned}
      profile={session.profile}
      sessionId={session.id}
      title={title}
    >
      <div
        data-session-id={session.id}
        draggable
        onDragEnd={() => onSessionDragEnd?.()}
        onDragStart={event => {
          const payload: SessionDragPayload = {
            archived,
            id: session.id,
            pinId: sessionPinId(session),
            pinned: isPinned,
            profile: session.profile || 'default',
            title
          }

          writeSessionDrag(event.dataTransfer, payload)
          onSessionDragStart?.(payload)
        }}
        ref={ref}
        style={style}
        {...rest}
      >
        <motion.div
          className={cn(
            'group relative grid min-h-[1.625rem] cursor-pointer grid-cols-[minmax(0,1fr)_auto] items-center rounded-md transition-[background-color,color,opacity,box-shadow] duration-100 ease-out hover:bg-(--ui-row-hover-background) hover:transition-none',
            reorderable && 'active:cursor-grabbing',
            (isSelected || checked) && 'bg-(--ui-row-active-background)',
            isWorking && 'text-foreground',
            dragging && 'z-10 cursor-grabbing opacity-60 shadow-sm',
            className
          )}
          data-actions-visible={actionsVisible ? 'true' : undefined}
          data-selected={checked ? 'true' : undefined}
          data-session-row-chrome
          data-working={isWorking ? 'true' : undefined}
          layout="position"
          onDoubleClick={selectionActive ? () => onResume() : undefined}
          transition={{ layout: { duration: 0.16, ease: [0.2, 0, 0, 1] } }}
        >
          {isWorking && !needsInput && <span aria-hidden="true" className="arc-border" />}
          <button
            className="z-0 flex min-w-0 items-center gap-1.5 bg-transparent py-0.5 pl-2 pr-2 text-left"
            onClick={event => {
              const canSelect = Boolean(selectable && onToggleSelect)

              // Desktop-convention selection on every selectable row:
              //   ⌘/⌃-click  — toggle THIS row in or out (non-contiguous sets,
              //                gaps welcome; also how a selection starts).
              //                Takes the binding over from open-in-new-window,
              //                which stays in the row's ⋯ / right-click menus.
              //   ⌥-click    — same toggle (legacy alias from the first cut).
              //   shift-click — contiguous range from the anchor; a cold
              //                shift-click seeds the anchor from the OPEN row
              //                so the run includes where the user started.
              //   plain click — resume normally; while a selection is active it
              //                toggles instead, and double-click still resumes.
              // Every gesture toggles, so re-clicking a selected row deselects
              // it regardless of modifier.
              if (canSelect && (event.metaKey || event.ctrlKey || event.altKey)) {
                event.preventDefault()
                event.stopPropagation()
                toggleSelect('single')

                return
              }

              if (canSelect && event.shiftKey) {
                event.preventDefault()
                event.stopPropagation()
                toggleSelect('range')

                return
              }

              if (canSelect && selectionActive) {
                event.preventDefault()
                event.stopPropagation()
                toggleSelect('single')

                return
              }

              // Rows outside any selectable section keep the legacy bindings.
              if (event.shiftKey) {
                event.preventDefault()
                event.stopPropagation()

                if (!archived) {
                  triggerHaptic('selection')
                  onPin()
                }

                return
              }

              // ⌘-click (mac) / ⌃-click (win/linux) pops the chat into its own
              // window on non-selectable rows. Falls through to a normal resume
              // when standalone windows aren't available (web embed).
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
            {selectionActive ? (
              // Selection mode: the leading dot column becomes the checkbox, so
              // rows don't shift horizontally when a selection starts (same
              // w-3.5 slot the other leading affordances use).
              <span aria-checked={checked} className="grid w-3.5 shrink-0 place-items-center" role="checkbox">
                <span
                  className={cn(
                    'grid size-3 place-items-center rounded-[3px] border transition-colors',
                    checked
                      ? 'border-foreground/80 bg-foreground/90 text-(--ui-sidebar-surface-background,var(--background))'
                      : 'border-(--ui-stroke-secondary) bg-transparent'
                  )}
                >
                  {checked && <Codicon name="check" size="0.5rem" />}
                </span>
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
              <span className="block truncate text-[0.8125rem] font-normal text-(--ui-text-secondary) group-hover:text-foreground group-data-[working=true]:text-foreground/90">
                {title}
              </span>
            </div>
          </button>
          {/* Trailing slot: on an IDLE row the timestamp is visible and slides
            left only while the row is hovered or the menu/keyboard action is
            actually visible. Plain pointer focus must not keep it displaced
            after the menu closes. On an ACTIVE row the pulsing orange dot on
            the left already signals "running", so the timestamp is hidden —
            but its width is still reserved (opacity-0, not unmounted) so the
            menu lands in the same spot and the row height never shifts.
            Transform/opacity only — no layout reflow. */}
          <div className="relative flex h-full items-center justify-end self-stretch pl-1 pr-1.5">
            <span
              className={cn(
                'pointer-events-none min-w-6 text-right text-[0.625rem] leading-none text-(--ui-text-tertiary) transition-[transform,opacity] duration-150 ease-out',
                // Slide left past the menu's footprint so the age stays fully
                // legible beside the revealed 3-dot button.
                'group-hover:-translate-x-6 group-data-[actions-visible=true]:-translate-x-6',
                // Active sessions: the orange dot is the status cue; hide the
                // timestamp (keep its reserved width) for the whole active run.
                // A clarify-blocked run keeps the timestamp visible because
                // the actionable state is "waiting on user", not live motion.
                isWorking && !needsInput && 'opacity-0'
              )}
              data-session-row-age
            >
              {age}
            </span>
            <div className="absolute inset-y-0 right-1 grid place-items-center">
              <SessionActionsMenu
                archived={archived}
                onArchive={onArchive}
                onDelete={onDelete}
                onOpenChange={setActionsMenuOpen}
                onPin={onPin}
                onRestore={onRestore}
                onSelect={selectable && onToggleSelect ? () => toggleSelect('single') : undefined}
                pinned={isPinned}
                profile={session.profile}
                sessionId={session.id}
                title={title}
              >
                <Button
                  aria-label={r.actionsFor(title)}
                  className="size-5 translate-x-1 scale-90 rounded-[4px] bg-transparent text-transparent opacity-0 transition-all duration-150 ease-out group-hover:translate-x-0 group-hover:scale-100 group-hover:text-(--ui-text-tertiary) group-hover:opacity-100 group-data-[actions-visible=true]:translate-x-0 group-data-[actions-visible=true]:scale-100 group-data-[actions-visible=true]:text-(--ui-text-tertiary) group-data-[actions-visible=true]:opacity-100 hover:bg-(--ui-control-active-background)! hover:text-foreground! focus-visible:translate-x-0 focus-visible:scale-100 focus-visible:bg-(--ui-control-active-background) focus-visible:text-foreground focus-visible:opacity-100 focus-visible:ring-0 data-[state=open]:translate-x-0 data-[state=open]:scale-100 data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground data-[state=open]:opacity-100 [&_svg]:size-3.5!"
                  data-session-row-actions
                  onBlur={() => setActionsKeyboardFocus(false)}
                  onFocus={event => setActionsKeyboardFocus(event.currentTarget.matches(':focus-visible'))}
                  size="icon"
                  title={r.sessionActions}
                  variant="ghost"
                >
                  <Codicon name="ellipsis" size="0.875rem" />
                </Button>
              </SessionActionsMenu>
            </div>
          </div>
        </motion.div>
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
