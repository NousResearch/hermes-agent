import { useStore } from '@nanostores/react'
import { memo } from 'react'
import type * as React from 'react'

import { PrTag } from '@/app/chat/pr-tag'
import { ProfileTag } from '@/app/chat/profile-tag'
import { startSessionDrag } from '@/app/chat/session-drag'
import { PlatformAvatar } from '@/app/messaging/platform-icon'
import { openSession } from '@/app/open-session'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { compactNumber } from '@/lib/format'
import { triggerHaptic } from '@/lib/haptics'
import { middleClickHandlers } from '@/lib/middle-click'
import { handoffOriginSource, sessionSourceLabel } from '@/lib/session-source'
import { coarseElapsed } from '@/lib/time'
import { useStoreSelector } from '@/lib/use-session-slice'
import { cn } from '@/lib/utils'
import { $sidebarRowMeta } from '@/store/layout'
import { normalizeProfileKey } from '@/store/profile'
import { $pullRequestsByBranch, sessionPrKey } from '@/store/pull-requests'
import { $sessionDotStateById, hasLiveTurn, showsRunningArc } from '@/store/session-dot-state'
import { sessionCostUsd } from '@/store/sidebar-archive'

import { SessionStatusDot } from '../session-status-dot'

import {
  SidebarRowBody,
  SidebarRowGrab,
  SidebarRowLabel,
  SidebarRowLead,
  SidebarRowLeadGlyph,
  SidebarRowShell
} from './chrome'
import { SessionActionsMenu, SessionContextMenu } from './session-actions-menu'
import { useProfilePrewarm } from './use-profile-prewarm'

interface SidebarSessionRowProps extends React.ComponentProps<'div'> {
  session: SessionInfo
  /** TUI-style tree stem for branched sessions (`└─ ` / `├─ `). */
  branchStem?: string
  isPinned: boolean
  isSelected: boolean
  onArchive: () => void
  onBranch?: () => void
  onDelete: () => void
  onPin: () => void
  onResume: () => void
  reorderable?: boolean
  dragging?: boolean
  dragHandleProps?: React.HTMLAttributes<HTMLElement>
  /** Tag the row with its owning profile (initial chip + tooltip). Used by
   *  flat cross-profile lists — Pinned and search results in the All-profiles
   *  view — where no group header communicates ownership (#66003). */
  showProfile?: boolean
}

const AGE_KEY = { day: 'ageDay', hour: 'ageHour', minute: 'ageMin' } as const

function formatAge(seconds: number, r: Translations['sidebar']['row']): string {
  const { unit, value } = coarseElapsed(Date.now() - seconds * 1000)

  // Under a minute reads as "now" — the sidebar never shows a seconds tick.
  return unit === 'second' ? r.ageNow : `${value}${r[AGE_KEY[unit]]}`
}

function SidebarSessionRowImpl({
  session,
  branchStem,
  isPinned,
  isSelected,
  onArchive,
  onBranch,
  onDelete,
  onPin,
  onResume,
  reorderable = false,
  dragging = false,
  dragHandleProps,
  showProfile = false,
  className,
  style,
  ref,
  ...rest
}: SidebarSessionRowProps) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const { cancelPrewarm, startPrewarm } = useProfilePrewarm(session.profile)
  const title = sessionTitle(session)
  const age = formatAge(session.last_active || session.started_at, r)
  const handleLabel = `Reorder ${title}`
  // Opt-in row metadata from the sidebar's filter menu. Read from the store
  // rather than threaded as props: the subscription re-renders past the memo
  // below, and a toggle should repaint every row at once anyway.
  const rowMeta = useStore($sidebarRowMeta)
  // Pinned metadata occupies the actions slot and swaps out for the kebab on
  // hover, so the row reserves the same width either way and never reflows.
  const pinnedAge = rowMeta.includes('updated')
  // The default profile has no mark worth spending a row slot on — a chip on
  // every row that says "the normal one" is noise. Named profiles only.
  const hasProfileTag = normalizeProfileKey(session.profile) !== 'default'
  const pinnedProfile = hasProfileTag && rowMeta.includes('profile')
  // The branch's PR, if the row was asked to show one. A selector, not a plain
  // useStore: a repo's PRs land as a single map write, and only the rows on
  // those branches should repaint.
  const prKey = sessionPrKey(session)
  const pr = useStoreSelector($pullRequestsByBranch, prs => (rowMeta.includes('pr') && prKey ? prs[prKey] : undefined))
  const totalTokens = session.input_tokens + session.output_tokens
  const cost = sessionCostUsd(session)

  // Tokens, cost and age share the one trailing slot rather than each claiming
  // their own column: several switched on read as one figure, not as a
  // widening gutter.
  const pinnedFacts = [
    rowMeta.includes('tokens') && totalTokens > 0 ? compactNumber(totalTokens) : null,
    // Sub-cent spend rounds to "$0.00", which reads as a bug rather than as a
    // cheap session — below a cent the row says nothing at all.
    rowMeta.includes('cost') && cost >= 0.01 ? `$${cost.toFixed(2)}` : null,
    pinnedAge ? age : null
  ].filter(Boolean) as string[]

  // The kebab covers the END of the slot, so only the last fact steps aside for
  // it. With tokens and age both on, hovering costs you the age and keeps the
  // number you switched on to read.
  const pinnedTail = pinnedFacts.at(-1) ?? ''
  const pinnedHead = pinnedFacts.slice(0, -1).join(' · ')
  const pinnedLabel = pinnedFacts.join(' · ')
  // Chips that ride in the BODY, beside the title. The kebab lifts out of the
  // actions slot, so it only ever covers what's in there — a body chip has no
  // reason to step aside, and the hover-age (which does overlay the body) is
  // dropped rather than made to fight them.
  const bodyChip = Boolean(pr) || pinnedProfile || (showProfile && hasProfileTag)
  const pinnedMeta = Boolean(pinnedLabel) || bodyChip
  // A handed-off session's live source is local, but it originated on a
  // messaging platform — surface that origin as a small badge so e.g. a
  // Telegram thread continued here still reads as Telegram.
  const handoffSource = handoffOriginSource(session.handoff_state, session.handoff_platform)
  const handoffLabel = handoffSource ? (sessionSourceLabel(handoffSource) ?? handoffSource) : null
  // The same resolved state the row's dot paints, so the arc and the dot cannot
  // contradict each other. A selector, not a plain useStore: the map is rebuilt
  // whenever any session's status changes, but a row only repaints on its own.
  const dotState = useStoreSelector($sessionDotStateById, states => states[session.id] ?? 'idle')
  const liveTurn = hasLiveTurn(dotState)

  // An archived session has no live status to paint, so the archive glyph takes
  // the lead slot the dot would occupy instead of adding a column of its own.
  const lead = session.archived ? (
    <SidebarRowLeadGlyph className="text-(--ui-text-quaternary)">
      <Codicon name="archive" size="0.75rem" />
    </SidebarRowLeadGlyph>
  ) : null

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
      <SidebarRowShell
        actions={
          // Pinned metadata sits in normal flow and the kebab lifts out of it,
          // so this slot's intrinsic width IS the metadata's — the row's
          // `auto` actions column measures it and the title truncates against
          // whatever is switched on, with no width to hand-maintain.
          <div className="relative z-2 flex items-center justify-end" data-row-actions>
            {/* Pinned metadata stays put through a turn — it was switched on to
                be read. Only the tail hands its slot to the kebab; anything
                ahead of it stays legible while you hover. The hover-only age is
                an overlay instead, so it needs the row to open up 48px of right
                padding — which beside a body chip reads as a hole. A row that
                already shows a chip skips it. */}
            {(pinnedLabel || (!liveTurn && !bodyChip)) && (
              <span
                className={cn(
                  'pointer-events-none whitespace-nowrap text-right text-[0.625rem] leading-none text-(--ui-text-tertiary)',
                  !pinnedLabel && 'absolute right-6 opacity-0 transition-opacity group-hover:opacity-100'
                )}
              >
                {pinnedLabel ? (
                  <>
                    {pinnedHead}
                    {/* Never narrower than the kebab that has to cover it. */}
                    <span className="inline-block min-w-5 transition-opacity group-hover:opacity-0">
                      {pinnedHead && ' · '}
                      {pinnedTail}
                    </span>
                  </>
                ) : (
                  age
                )}
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
                aria-label={r.sessionActions}
                className={cn(
                  'size-5 rounded-[4px] bg-transparent text-transparent transition-colors duration-100 hover:bg-(--ui-control-active-background) hover:text-foreground focus-visible:bg-(--ui-control-active-background) focus-visible:text-foreground focus-visible:ring-0 data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground group-hover:text-(--ui-text-tertiary) [&_svg]:size-3.5!',
                  pinnedLabel && 'absolute right-0'
                )}
                size="icon"
                variant="ghost"
              >
                <Codicon name="kebab-vertical" size="0.875rem" />
              </Button>
            </SessionActionsMenu>
          </div>
        }
        className={cn(
          'group row-hover relative',
          isSelected && 'bg-(--ui-row-active-background)',
          liveTurn && 'text-foreground',
          // Opaque surface while lifted so the dragged row erases what's under
          // it (translucency let the rows below bleed through).
          dragging && 'z-10 cursor-grabbing bg-(--ui-sidebar-surface-background)',
          className
        )}
        data-working={liveTurn ? 'true' : undefined}
        // The row runs BOTH drags off one press, and each declines outside its
        // own region — so no timing/arbitration rule is needed and neither can
        // steal the other's gesture. Over the sidebar only the reorder has a
        // target (the session drop denies: side chrome hosts no main tile);
        // over the tree only the session drop does (no sortable row there).
        // Whichever one the release lands on is the one that commits.
        {...dragHandleProps}
        onPointerDown={event => {
          // The grabber already carries these same listeners, and the ⋯
          // cluster keeps its own gestures.
          if ((event.target as HTMLElement).closest('[data-reorder-handle], [data-row-actions]')) {
            return
          }

          // A POINTER drag on the shared drag session (never native HTML5 DnD:
          // no macOS snap-back, Esc aborts instantly). Sub-threshold releases
          // stay ordinary clicks, so resume / pin / open-in-window are
          // untouched.
          startSessionDrag({ id: session.id, profile: session.profile || 'default', title }, event)
          dragHandleProps?.onPointerDown?.(event)
        }}
        // Hovering a row from another profile (the all-profiles view) telegraphs
        // a cross-profile resume — start that backend's spawn now so the click
        // doesn't pay the full cold boot. Same-profile rows no-op inside
        // prewarmProfileBackend.
        onPointerEnter={startPrewarm}
        onPointerLeave={cancelPrewarm}
        ref={ref}
        style={style}
        {...rest}
      >
        {showsRunningArc(dotState) && <span aria-hidden="true" className="arc-border arc-row" />}
        <SidebarRowBody
          // Pinned metadata already sits in the actions slot, so the title only
          // needs a gap from it — and the same gap on hover, or the row would
          // jump every time the kebab took over.
          className={cn('z-0', pinnedMeta ? 'pr-2' : 'group-hover:pr-12', branchStem && 'pl-3.5')}
          // Middle-click = open in a new tab (browser muscle memory).
          {...middleClickHandlers(() => {
            triggerHaptic('selection')
            openSession(session.id, () => undefined, 'tab')
          })}
          onClick={event => {
            const mod = event.metaKey || event.ctrlKey

            // ⇧⌘-click → pop into its own window (needs standalone windows).
            if (mod && event.shiftKey) {
              event.preventDefault()
              event.stopPropagation()
              triggerHaptic('selection')
              openSession(session.id, () => undefined, 'window')

              return
            }

            // ⌘/⌃-click → open in a new tab (stack into main).
            if (mod) {
              event.preventDefault()
              event.stopPropagation()
              triggerHaptic('selection')
              openSession(session.id, () => undefined, 'tab')

              return
            }

            // ⇧-click → pin.
            if (event.shiftKey) {
              event.preventDefault()
              event.stopPropagation()
              triggerHaptic('selection')
              onPin()

              return
            }

            onResume()
          }}
        >
          {reorderable ? (
            <SidebarRowGrab ariaLabel={handleLabel} dragging={dragging} dragHandleProps={dragHandleProps}>
              {lead ?? (
                <SessionStatusDot
                  branchStem={branchStem}
                  className="transition-opacity group-hover/handle:opacity-0 group-focus-within/handle:opacity-0"
                  session={session}
                  storedSessionId={session.id}
                />
              )}
            </SidebarRowGrab>
          ) : (
            <SidebarRowLead className="overflow-hidden">
              {lead ?? <SessionStatusDot branchStem={branchStem} session={session} storedSessionId={session.id} />}
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
          {/* Stays put on hover, unlike the other chips: it's a link, and the
              kebab lives in its own column rather than over this one. */}
          {pr && <PrTag pr={pr} />}
          {(showProfile || pinnedProfile) && hasProfileTag && <ProfileTag profile={session.profile} />}
        </SidebarRowBody>
      </SidebarRowShell>
    </SessionContextMenu>
  )
}

// The sidebar re-renders on every stream tick ($sessions/$workingSessionIds
// churn), and it stays mounted beneath every overlay — so an unmemoized row
// re-rendered the whole list (and its Codicon/label/status-dot subtree) on each
// delta, bleeding churn into Settings, Cron, Profiles, Artifacts, etc.
//
// The callback props (onArchive/onResume/…) are fresh closures every render by
// design (they close over the row's session id), so a default memo never bails.
// They're pure id-forwarders, though — identical behavior for a given row — so
// the comparator deliberately ignores them and compares only the DATA that
// changes what the row paints. A row whose session/selection/pin state is
// unchanged now bails out, even while a sibling session streams; its own status
// arrives through a store subscription, which re-renders it past this bail.
function rowPropsEqual(a: SidebarSessionRowProps, b: SidebarSessionRowProps): boolean {
  return (
    a.session === b.session &&
    a.isPinned === b.isPinned &&
    a.isSelected === b.isSelected &&
    a.branchStem === b.branchStem &&
    a.reorderable === b.reorderable &&
    a.dragging === b.dragging &&
    a.showProfile === b.showProfile &&
    a.dragHandleProps === b.dragHandleProps &&
    a.className === b.className &&
    a.style === b.style
  )
}

export const SidebarSessionRow = memo(SidebarSessionRowImpl, rowPropsEqual)
