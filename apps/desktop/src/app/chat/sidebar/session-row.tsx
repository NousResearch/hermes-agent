import { useStore } from '@nanostores/react'
import { memo } from 'react'
import type * as React from 'react'

import { PrTag } from '@/app/chat/pr-tag'
import { ProfileTag } from '@/app/chat/profile-tag'
import { startSessionDrag } from '@/app/chat/session-drag'
import { PlatformAvatar } from '@/app/messaging/platform-icon'
import { openSession } from '@/app/open-session'
import { formatMessageTimestamp } from '@/components/assistant-ui/thread/timestamp'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { OverflowTip, Tip } from '@/components/ui/tooltip'
import type { SessionInfo } from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { pathLeaf } from '@/lib/display-path'
import { compactNumber } from '@/lib/format'
import { triggerHaptic } from '@/lib/haptics'
import { middleClickHandlers } from '@/lib/middle-click'
import { displayModelName } from '@/lib/model-status-label'
import { sessionProjectLabel } from '@/lib/session-project-label'
import { handoffOriginSource, sessionSourceLabel } from '@/lib/session-source'
import { coarseElapsed } from '@/lib/time'
import { useStoreSelector } from '@/lib/use-session-slice'
import { cn } from '@/lib/utils'
import { $sidebarRowMeta } from '@/store/layout'
import { normalizeProfileKey } from '@/store/profile'
import { $projects } from '@/store/projects'
import { $pullRequestsByBranch, sessionPrKey } from '@/store/pull-requests'
import { $sessionDotStateById, hasLiveTurn, showsRunningArc } from '@/store/session-dot-state'
import { $sessionListDensity } from '@/store/session-list-density'
import { $openStoredSessionIds } from '@/store/session-states'
import { sessionCostUsd } from '@/store/sidebar-archive'
import { $todoProgressBySession } from '@/store/todos'

import { SessionStatusDot } from '../session-status-dot'

import {
  SIDEBAR_ROW_CARD_MIN_H,
  SIDEBAR_TRUNCATED_LEADING,
  SidebarRowBody,
  SidebarRowGrab,
  SidebarRowLabel,
  SidebarRowLead,
  SidebarRowLeadGlyph,
  SidebarRowShell
} from './chrome'
import { SessionActionsMenu, SessionContextMenu } from './session-actions-menu'
import { sessionRowDetails } from './session-row-details'
import { resolveSessionRowClick } from './session-row-gesture'
import { useProfilePrewarm } from './use-profile-prewarm'

interface SidebarSessionRowProps extends React.ComponentProps<'div'> {
  session: SessionInfo
  /** TUI-style tree stem for branched sessions (`└─ ` / `├─ `). */
  branchStem?: string
  isPinned: boolean
  isSelected: boolean
  /** Backend-derived read state — same value the dot paints. */
  unread: boolean
  onArchive: () => void
  onBranch?: () => void
  onDelete: () => void
  onPin: () => void
  /** Toggle the persisted read-state watermark. */
  onToggleUnread: () => void
  onResume: () => void
  reorderable?: boolean
  dragging?: boolean
  dragHandleProps?: React.HTMLAttributes<HTMLElement>
  /** Tag the row with its owning profile (initial chip + tooltip). Used by
   *  flat cross-profile lists in the All-profiles view where no profile group
   *  header communicates ownership. */
  showProfile?: boolean
  /** Inbox-style card: workspace header, title + last-message preview, and a
   *  model · size footer. The flat recents list opts in via the filter menu;
   *  dense tree surfaces (projects, messaging, pins) keep the one-line row. */
  card?: boolean
}

const AGE_KEY = { day: 'ageDay', hour: 'ageHour', minute: 'ageMin' } as const

// Hover marquee (card title): measure the actual overflow on pointerenter and
// arm the CSS animation only when there is some — CSS can't detect overflow on
// its own, and animating a non-overflowing title would wiggle for nothing.
// Distance-proportional duration keeps the scroll speed constant across short
// and long overflows. State lives in DOM attributes, not React state: hover
// must not re-render a memoized row.
const MARQUEE_PX_PER_SECOND = 80

function armMarquee(event: React.PointerEvent<HTMLElement>) {
  const el = event.currentTarget
  const distance = el.scrollWidth - el.clientWidth

  if (distance > 2) {
    // The keyframes spend 65% of the cycle travelling (10%→75%); scale the
    // duration so the travel segment itself moves at the target speed.
    el.style.setProperty('--marquee-d', `${distance}px`)
    el.style.setProperty('--marquee-t', `${Math.max(1, distance / MARQUEE_PX_PER_SECOND / 0.65)}s`)
    el.dataset.marquee = 'true'
  }
}

function disarmMarquee(event: React.PointerEvent<HTMLElement>) {
  delete event.currentTarget.dataset.marquee
}

// The last thing in the trailing slot hands its place to the ⋯ button on hover,
// and is never narrower than the button that has to cover it. A PR chip is the
// exception while the pointer is on it: it's a link, and the kebab sits
// absolute over this space, so it has to stop taking clicks too, not just fade.
const TAIL_HIDES = 'min-w-5 transition-opacity group-hover:opacity-0 group-has-[[data-pr-link]:hover]:opacity-100'
const KEBAB_YIELDS = 'group-has-[[data-pr-link]:hover]:pointer-events-none group-has-[[data-pr-link]:hover]:opacity-0'

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
  unread,
  onArchive,
  onBranch,
  onDelete,
  onPin,
  onToggleUnread,
  onResume,
  reorderable = false,
  dragging = false,
  dragHandleProps,
  showProfile = false,
  card = false,
  className,
  style,
  ref,
  ...rest
}: SidebarSessionRowProps) {
  const { t } = useI18n()
  const r = t.sidebar.row
  const { cancelPrewarm, startPrewarm } = useProfilePrewarm(session.profile)
  const title = sessionTitle(session)
  const density = useStore($sessionListDensity)
  const fmt = t.sidebar

  const details = sessionRowDetails(session, {
    messageCount: fmt.messageCount,
    toolCallCount: fmt.toolCallCount
  })

  const timestamp = session.last_active || session.started_at
  const age = formatAge(timestamp, r)
  const timestampDate = new Date(timestamp * 1000)
  const absoluteAge = formatMessageTimestamp(timestampDate, t.assistant.thread)
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
  // Open in a pane, but not the focused one. A selector rather than a prop:
  // it reaches all four row render paths at once, the set only changes when a
  // tile opens or closes, and the boolean bails every unaffected row out.
  const openUnfocused = useStoreSelector($openStoredSessionIds, open => !isSelected && open.has(session.id))
  const totalTokens = session.input_tokens + session.output_tokens
  const cost = sessionCostUsd(session)

  // Tokens, cost and age share one figure rather than each claiming a column:
  // several switched on read as one number, not as a widening gutter.
  const figures = [
    rowMeta.includes('tokens') && totalTokens > 0 ? compactNumber(totalTokens) : null,
    // Sub-cent spend rounds to "$0.00", which reads as a bug rather than as a
    // cheap session — below a cent the row says nothing at all.
    rowMeta.includes('cost') && cost >= 0.01 ? `$${cost.toFixed(2)}` : null
  ].filter(Boolean) as string[]

  // Everything the Show menu puts after the title shares ONE right-aligned
  // slot, in reading order: identity chips, then the figures. The kebab covers
  // the END of that slot on hover, so only the last thing in it steps aside —
  // with tokens and age both on you lose the age and keep the number you
  // switched on, and a PR keeps its place (and its click) unless it IS the last
  // thing. Chips used to render in the body instead, which left them stranded
  // to the left of the kebab's own column: never flush right, never swapping.
  const trailing: { key: string; node: React.ReactNode }[] = []

  if ((showProfile || pinnedProfile) && hasProfileTag) {
    trailing.push({ key: 'profile', node: <ProfileTag profile={session.profile} /> })
  }

  if (pr) {
    trailing.push({ key: 'pr', node: <PrTag pr={pr} /> })
  }

  const showAge = pinnedAge || card

  if (figures.length || showAge) {
    // The card's meta lines separate by spacing alone, so its header figures
    // match (non-breaking pair — plain spaces collapse to one); the one-line
    // row keeps the interpunct between joined figures.
    const sep = card ? '\u00A0\u00A0' : ' · '
    const head = (showAge ? figures : figures.slice(0, -1)).join(sep)

    trailing.push({
      key: 'figures',
      node: (
        <span className="pointer-events-none whitespace-nowrap text-[0.625rem] leading-none text-(--ui-text-tertiary)">
          {head}
          {/* The figures own their tail: the separator goes with it. */}
          <span className={cn('inline-block text-right', TAIL_HIDES)}>
            {head && sep}
            {showAge ? (
              <Tip label={absoluteAge} side="top">
                <time
                  aria-label={`${age}, ${absoluteAge}`}
                  className="pointer-events-auto focus-visible:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-sidebar-ring"
                  dateTime={timestampDate.toISOString()}
                  tabIndex={0}
                >
                  {age}
                </time>
              </Tip>
            ) : (
              figures.at(-1)
            )}
          </span>
        </span>
      )
    })
  }

  // A chip that ends the slot hides whole; the figures handle their own tail.
  const chipEndsSlot = trailing.length > 0 && !figures.length && !pinnedAge
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

  // Card header line: the workspace this belongs to — the project when it
  // resolves (same function the session color reads, so name and tint agree;
  // a worktree reports its repo, not the scratch dir it sits in), else the
  // bare cwd leaf, else the same synthetic "Home" the project views use for
  // workspace-less chats. Always text: an empty header line reads as a hole.
  // A SELECTOR, not useStore($projects): the projects atom refreshes on the
  // tree poll with fresh identity, and a plain subscription would re-render
  // every row (card or not) on every poll. Selecting the resolved label means
  // a row only repaints when its own label actually changes — and one-line
  // rows always select null.
  const context = useStoreSelector($projects, projects =>
    card ? (sessionProjectLabel(session, projects) ?? (pathLeaf(session.cwd) || t.sidebar.projects.home)) : null
  )

  // Card footer line: which model worked on it and how big it got. Rendered
  // as separate spans with a flex gap — a joined string can't put real space
  // between them (HTML collapses runs of whitespace to one).
  const model = card && session.model ? displayModelName(session.model) : ''
  const size = card && session.message_count > 0 ? r.messageCount(session.message_count) : ''
  // Live plan progress ("3/7"), far right of the footer. A selector keyed to
  // this row: only rows whose own fraction changes repaint on todo events.
  const todoProgress = useStoreSelector($todoProgressBySession, progress => (card ? progress[session.id] : undefined))

  // An archived session has no live status to paint, so the archive glyph takes
  // the lead slot the dot would occupy instead of adding a column of its own.
  const lead = session.archived ? (
    <SidebarRowLeadGlyph className="text-(--ui-text-quaternary)">
      <Codicon name="archive" size="0.75rem" />
    </SidebarRowLeadGlyph>
  ) : null

  // The trailing metadata sits in normal flow and the kebab lifts out of it,
  // so this cluster's intrinsic width IS the metadata's. In the one-line row
  // it rides the shell's `auto` actions column and the title truncates
  // against it. In the card it renders INSIDE the header row instead — the
  // shell column would span the card's full height and shave every line,
  // when only the header shares its line with the age and kebab.
  const actionsNode = (
    <div className="relative z-2 flex shrink-0 items-center justify-end gap-1" data-row-actions>
      {trailing.map(({ key, node }, index) => (
        <span
          className={
            chipEndsSlot && index === trailing.length - 1 ? cn('inline-flex justify-end', TAIL_HIDES) : undefined
          }
          key={key}
        >
          {node}
        </span>
      ))}
      <SessionActionsMenu
        onArchive={onArchive}
        onBranch={onBranch}
        onDelete={onDelete}
        onPin={onPin}
        onToggleUnread={onToggleUnread}
        pinned={isPinned}
        profile={session.profile}
        sessionId={session.id}
        title={title}
        unread={unread}
      >
        <Button
          aria-label={r.sessionActions}
          className={cn(
            'size-5 rounded-[4px] bg-transparent text-transparent transition-colors duration-100 hover:bg-(--ui-control-active-background) hover:text-foreground focus-visible:bg-(--ui-control-active-background) focus-visible:text-foreground focus-visible:ring-0 data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground group-hover:text-(--ui-text-tertiary) [&_svg]:size-3.5!',
            trailing.length > 0 && 'absolute right-0',
            pr && KEBAB_YIELDS
          )}
          size="icon"
          variant="ghost"
        >
          <Codicon name="kebab-vertical" size="0.875rem" />
        </Button>
      </SessionActionsMenu>
    </div>
  )

  return (
    <SessionContextMenu
      onArchive={onArchive}
      onBranch={onBranch}
      onDelete={onDelete}
      onPin={onPin}
      onToggleUnread={onToggleUnread}
      pinned={isPinned}
      profile={session.profile}
      sessionId={session.id}
      title={title}
      unread={unread}
    >
      <SidebarRowShell
        actions={card ? undefined : actionsNode}
        className={cn(
          'group row-hover relative',
          card && SIDEBAR_ROW_CARD_MIN_H,
          // Density-aware minimum heights for the inline (non-card) row: the
          // metadata / preview lines below need the extra rows (#68119).
          !card && density !== 'compact' && 'min-h-[2.75rem]',
          !card && density === 'detailed' && 'min-h-[3.875rem]',
          isSelected && 'bg-(--ui-row-active-background)',
          // Open in another pane: the SAME band, just weaker. Its own mixed
          // token rather than row opacity — dimming the whole row would take
          // the title and the status dot down with it.
          openUnfocused && 'bg-(--ui-row-open-background)',
          liveTurn && 'text-foreground',
          // Opaque surface while lifted so the dragged row erases what's under
          // it (translucency let the rows below bleed through). data-glass-opaque
          // keeps that true when window glass thins the field.
          dragging && 'z-10 cursor-grabbing bg-(--ui-sidebar-surface-background)',
          className
        )}
        data-glass-opaque={dragging ? '' : undefined}
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
          // Every trailing figure lives in the actions slot, which the row
          // measures — so the title needs a gap from it and nothing else. Hover
          // changes what you can see in that slot, never how wide it is. The
          // card has no such column to clear (its cluster is INSIDE the body,
          // ending at the shell's own trailing inset), and keeping the gap
          // would pull the header in past every line below it.
          className={cn(
            'z-0',
            card && 'pr-0',
            branchStem && 'pl-3.5',
            // The card is a grid with ONE spacing knob: --card-gap. Every row
            // gap is gap-y-(--card-gap); the title/preview group opts out
            // with its own tighter internal flex gap.
            card && 'flex-col items-stretch justify-center py-1.5 [--card-gap:0.4rem] gap-(--card-gap)'
          )}
          // Middle-click = open in a new tab (browser muscle memory).
          {...middleClickHandlers(() => {
            triggerHaptic('selection')
            openSession(session.id, () => undefined, 'tab')
          })}
          onClick={event => {
            // Modifier-click gestures on a row (see `resolveSessionRowClick`):
            //   ⇧          → pin / unpin
            //   ⌘/⌃        → open in a new tab (stack into main)
            //   ⌘/⌃ + ⇧    → pop into its own window (needs standalone windows)
            //   ⌥ + ⇧      → archive
            // A plain click resumes. Archive also lives in the row's ⋯ and
            // right-click menus and as a rebindable hotkey (`session.archive`).
            // `openSession`'s 'window' intent already falls back to 'tab' when
            // the bridge lacks standalone windows, so the resolver can always
            // offer the window action here.
            const action = resolveSessionRowClick(event, { canOpenWindow: true })

            if (action === 'resume') {
              onResume()

              return
            }

            event.preventDefault()
            event.stopPropagation()
            triggerHaptic('selection')

            if (action === 'archive') {
              onArchive()
            } else if (action === 'pin') {
              onPin()
            } else if (action === 'newTab') {
              openSession(session.id, () => undefined, 'tab')
            } else {
              openSession(session.id, () => undefined, 'window')
            }
          }}
        >
          {(() => {
            const leadNode = reorderable ? (
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
            )

            const handoffBadge =
              handoffSource && handoffLabel ? (
                <Tip label={r.handoffOrigin(handoffLabel)}>
                  <PlatformAvatar
                    className="-mt-px size-4 shrink-0 rounded-[4px] text-[0.5rem] [&_svg]:size-2.5"
                    platformId={handoffSource}
                    platformName={handoffLabel}
                  />
                </Tip>
              ) : null

            if (!card) {
              return (
                <>
                  {leadNode}
                  {handoffBadge}
                  <span className="min-w-0 flex-1 self-center">
                    <OverflowTip label={title}>
                      <SidebarRowLabel
                        className="hover-marquee block font-normal group-hover:text-foreground group-data-[working=true]:text-foreground/90"
                        onPointerEnter={armMarquee}
                        onPointerLeave={disarmMarquee}
                      >
                        <span className="hover-marquee-inner">{title}</span>
                      </SidebarRowLabel>
                    </OverflowTip>
                    {/* Session-list density (#68119): comfortable adds one
                        deterministic metadata line; detailed adds the initial
                        request preview. Compact keeps today's one-line row. */}
                    {density !== 'compact' && details.metadata && (
                      <span
                        className={cn(
                          'mt-0.5 block truncate text-[0.625rem] text-(--ui-text-tertiary)',
                          SIDEBAR_TRUNCATED_LEADING
                        )}
                      >
                        {details.metadata}
                      </span>
                    )}
                    {density === 'detailed' && details.preview && (
                      <span
                        className={cn(
                          'mt-1 block truncate text-[0.625rem] text-(--ui-text-quaternary)',
                          SIDEBAR_TRUNCATED_LEADING
                        )}
                      >
                        {details.preview}
                      </span>
                    )}
                  </span>
                </>
              )
            }

            return (
              <>
                {/* Header row — ONE div: dot, context, then the age/kebab
                    cluster in flow at its right edge. Keeping the cluster
                    inside this line (instead of the shell's full-height side
                    column) means title/preview/meta below span the card's
                    entire width — nothing truncates against the kebab. */}
                <div className="flex min-w-0 items-center gap-1.5">
                  {leadNode}
                  <span
                    className={cn(
                      'min-w-0 flex-1 truncate text-[0.6875rem] text-(--ui-text-tertiary)',
                      SIDEBAR_TRUNCATED_LEADING
                    )}
                  >
                    {context}
                  </span>
                  {handoffBadge}
                  {actionsNode}
                </div>
                {/* Title + preview: ONE grouped cell with its own tight
                    internal gap — it does not inherit the card's rhythm. */}
                <div className="flex min-w-0 flex-col gap-[0.15rem]">
                  <OverflowTip label={title}>
                    <SidebarRowLabel
                      className={cn(
                        'hover-marquee text-[0.8125rem] font-medium text-(--ui-text-primary) group-data-[working=true]:text-foreground',
                        SIDEBAR_TRUNCATED_LEADING
                      )}
                      onPointerEnter={armMarquee}
                      onPointerLeave={disarmMarquee}
                    >
                      <span className="hover-marquee-inner">{title}</span>
                    </SidebarRowLabel>
                  </OverflowTip>
                  {session.preview && rowMeta.includes('preview') ? (
                    <span
                      className={cn(
                        'min-w-0 truncate text-[0.625rem] text-(--ui-text-quaternary)',
                        SIDEBAR_TRUNCATED_LEADING
                      )}
                    >
                      {session.preview}
                    </span>
                  ) : null}
                </div>
                {model || size || todoProgress ? (
                  <span
                    className={cn(
                      'flex min-w-0 items-baseline gap-2 text-[0.625rem] text-(--ui-text-tertiary)',
                      SIDEBAR_TRUNCATED_LEADING
                    )}
                  >
                    {model ? <span className="min-w-0 truncate">{model}</span> : null}
                    {size ? <span className="shrink-0 tabular-nums">{size}</span> : null}
                    {todoProgress ? (
                      <span className="ml-auto shrink-0 tabular-nums" title={r.todoProgress}>
                        {todoProgress}
                      </span>
                    ) : null}
                  </span>
                ) : null}
              </>
            )
          })()}
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
    a.unread === b.unread &&
    a.branchStem === b.branchStem &&
    a.reorderable === b.reorderable &&
    a.dragging === b.dragging &&
    a.showProfile === b.showProfile &&
    a.card === b.card &&
    a.dragHandleProps === b.dragHandleProps &&
    a.className === b.className &&
    a.style === b.style
  )
}

export const SidebarSessionRow = memo(SidebarSessionRowImpl, rowPropsEqual)
