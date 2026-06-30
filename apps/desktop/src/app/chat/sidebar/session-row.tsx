import { useStore } from '@nanostores/react'
import type * as React from 'react'
import { useRef, useState } from 'react'

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
import { $attentionSessionIds } from '@/store/session'
import { canOpenSessionWindow, openSessionInNewWindow } from '@/store/windows'

import { SidebarRowBody, SidebarRowGrab, SidebarRowLabel, SidebarRowLead, SidebarRowShell } from './chrome'
import { SessionActionsMenu, SessionContextMenu } from './session-actions-menu'

interface SidebarSessionRowProps extends React.ComponentProps<'div'> {
  session: SessionInfo
  /** TUI-style tree stem for branched sessions (`└─ ` / `├─ `). */
  branchStem?: string
  isPinned: boolean
  isSelected: boolean
  isWorking: boolean
  onArchive: () => void
  onBranch?: () => void
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
  branchStem,
  isPinned,
  isSelected,
  isWorking,
  onArchive,
  onBranch,
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
  const handoffLabel = handoffSource ? (sessionSourceLabel(handoffSource) ?? handoffSource) : null
  // Subscribe per-row (the leaf) instead of drilling a set through the list —
  // the atom is tiny and rarely non-empty. True when a clarify prompt in this
  // session is waiting on the user.
  const needsInput = useStore($attentionSessionIds).includes(session.id)

  // Mobile gesture handler: long-press = pin (replaces the desktop's shift-click,
  // which a touch device can't produce). On mobile we also drop the Radix
  // ContextMenu wrapper so its long-press menu doesn't race the tap and let
  // the user accidentally hit "Pin" as the first item.
  // Read both the global flag AND the html class — the flag is set in
  // injectedJavaScriptBeforeContentLoaded but during HMR there's a window
  // where a fresh module re-evaluates before main.tsx runs again. classList
  // is set once at boot and never removed, so it's the more reliable gate.
  const [mobileStandalone] = useState(
    () =>
      typeof window !== 'undefined' &&
      (Boolean((window as { __HERMES_MOBILE_STANDALONE__?: boolean }).__HERMES_MOBILE_STANDALONE__) ||
        (typeof document !== 'undefined' && document.documentElement.classList.contains('hermes-mobile-standalone')) ||
        'ontouchstart' in window)
  )
  // Long-press → pin. The previous timer-based approach fired onPin from a
  // setTimeout regardless of what the finger did afterward. On iOS WebKit,
  // when the user touched the row and then started scrolling the drawer,
  // pointerup/pointercancel were NOT always dispatched on the row — WebKit
  // routes the gesture to the scroll container and eats the cancel. The
  // timer kept ticking and fired onPin even though the user was just
  // scrolling, so every tap-and-scroll pinned the row.
  //
  // New approach: gate onPin on RELEASE, not on a timer. Track press start
  // time + origin in a ref; on pointerup, if duration ≥ 700ms AND finger
  // didn't move > 8px, fire onPin. If WebKit eats the up (scroll), the
  // pressInfo just leaks until the next press overwrites it — no timer
  // means no spurious fire.
  const longPressFired = useRef(false)
  const pressInfo = useRef<{ startedAt: number; x: number; y: number } | null>(null)
  const clearPress = () => {
    pressInfo.current = null
  }
  const onPointerDown = (event: React.PointerEvent<HTMLButtonElement>) => {
    if (!mobileStandalone || event.pointerType !== 'touch') return
    longPressFired.current = false
    pressInfo.current = { startedAt: Date.now(), x: event.clientX, y: event.clientY }
  }
  const onPointerMove = (event: React.PointerEvent<HTMLButtonElement>) => {
    const info = pressInfo.current
    if (!info) return
    const dx = event.clientX - info.x
    const dy = event.clientY - info.y
    // > 8px in either axis = scroll/drag, not a hold-to-pin. Clear so the
    // up event can't fire onPin even if the dwell time crossed 700ms.
    if (dx * dx + dy * dy > 64) clearPress()
  }
  const onPointerUp = (event: React.PointerEvent<HTMLButtonElement>) => {
    const info = pressInfo.current
    clearPress()
    if (!mobileStandalone || event.pointerType !== 'touch' || !info) return
    if (Date.now() - info.startedAt >= 700) {
      longPressFired.current = true
      triggerHaptic('selection')
      onPin()
    }
  }
  const onPointerUpOrLeave = () => clearPress()

  const shell = (
    <SidebarRowShell
      data-session-row=""
      actions={
        // Mobile-standalone: don't render the kebab DropdownMenu at all. The
        // CSS display:none hide was racing with first paint, leaving a Radix
        // Slot trigger area tappable on touch; once it received a stray tap
        // the Pin item (first in the menu) fired immediately. Mobile has
        // long-press = pin and (eventually) a Done/edit overlay instead.
        mobileStandalone ? null : (
          <div className="relative z-2 grid w-[1.375rem] place-items-center">
            {!isWorking && (
              <span className="pointer-events-none absolute right-6 top-1/2 min-w-6 -translate-y-1/2 text-right text-[0.625rem] leading-none text-(--ui-text-tertiary) opacity-0 transition-opacity group-hover:opacity-100">
                {age}
              </span>
            )}
            <SessionActionsMenu
              onArchive={onArchive}
              onBranch={onBranch}
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
                data-no-tap-min=""
                data-row-kebab=""
                size="icon"
                title={r.sessionActions}
                variant="ghost"
              >
                <Codicon name="kebab-vertical" size="0.875rem" />
              </Button>
            </SessionActionsMenu>
          </div>
        )
      }
      className={cn(
        'group relative cursor-pointer transition-colors duration-100 ease-out hover:bg-(--ui-row-hover-background) hover:transition-none',
        isSelected && 'bg-(--ui-row-active-background)',
        isWorking && 'text-foreground',
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
      <SidebarRowBody
        className={cn('z-0 group-hover:pr-12', branchStem && 'pl-3.5')}
        onClick={event => {
          // Mobile long-press already fired onPin; suppress the synthetic
          // click that follows the pointer-up so we don't also resume.
          if (longPressFired.current) {
            longPressFired.current = false
            event.preventDefault()
            event.stopPropagation()
            return
          }
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
        onPointerCancel={onPointerUpOrLeave}
        onPointerDown={onPointerDown}
        onPointerLeave={onPointerUpOrLeave}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        {reorderable ? (
          <SidebarRowGrab
            ariaLabel={handleLabel}
            dragging={dragging}
            dragHandleProps={dragHandleProps}
            leadClassName={needsInput ? 'overflow-visible' : undefined}
          >
            <SessionRowLeadDot
              branchStem={branchStem}
              className="transition-opacity group-hover/handle:opacity-0 group-focus-within/handle:opacity-0"
              isWorking={isWorking}
              needsInput={needsInput}
            />
          </SidebarRowGrab>
        ) : (
          <SidebarRowLead className={needsInput ? 'overflow-visible' : 'overflow-hidden'}>
            <SessionRowLeadDot branchStem={branchStem} isWorking={isWorking} needsInput={needsInput} />
          </SidebarRowLead>
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
        <SidebarRowLabel className="flex-1 font-normal group-hover:text-foreground group-data-[working=true]:text-foreground/90">
          {title}
        </SidebarRowLabel>
      </SidebarRowBody>
    </SidebarRowShell>
  )

  if (mobileStandalone) return shell

  return (
    <SessionContextMenu
      onArchive={onArchive}
      onBranch={onBranch}
      onDelete={onDelete}
      onPin={onPin}
      pinned={isPinned}
      profile={session.profile}
      sessionId={session.id}
      title={title}
    >
      {shell}
    </SessionContextMenu>
  )
}

function SessionRowLeadDot({
  branchStem,
  isWorking,
  needsInput = false,
  className
}: {
  branchStem?: string
  isWorking: boolean
  needsInput?: boolean
  className?: string
}) {
  return (
    <span className={cn('flex items-center gap-0.5', className)}>
      {branchStem ? (
        <span aria-hidden className="shrink-0 font-mono text-[0.625rem] leading-none text-(--ui-text-quaternary)">
          {branchStem}
        </span>
      ) : null}
      <SidebarRowDot isWorking={isWorking} needsInput={needsInput} />
    </span>
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
          ? "relative size-1.5 bg-(--ui-accent) shadow-[0_0_0.625rem_color-mix(in_srgb,var(--ui-accent)_55%,transparent)] before:absolute before:inset-0 before:animate-ping before:rounded-full before:bg-(--ui-accent) before:opacity-70 before:content-['']"
          : 'size-1 bg-(--ui-text-quaternary) opacity-80',
        className
      )}
      role={isWorking ? 'status' : undefined}
    />
  )
}
