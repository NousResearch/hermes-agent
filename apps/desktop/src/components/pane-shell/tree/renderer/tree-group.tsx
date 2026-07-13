/**
 * Group node renderer — a ZONE: header strip (tabs when stacked, minimize
 * chevron) + the active pane's content, resolved from the contribution
 * registry (`area: 'panes'`). Empty zones exist only in editor-authored
 * trees (drop targets until the first structural op prunes them).
 *
 * Dragging is FancyZones-style (drag-session.ts): the layout stays fixed and
 * every zone lights up as a whole-region drop target. Right-click opens the
 * contextual zone menu (split/move + header/minimize toggles).
 */

import { useStore } from '@nanostores/react'
import { type CSSProperties, Fragment, type ReactNode, type RefObject, useRef, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from '@/components/ui/context-menu'
import { DecodeText } from '@/components/ui/decode-text'
import { DROP_SHEET_BLUR_CLASS, DROP_SHEET_CLASS, DropPill } from '@/components/ui/drop-affordance'
import { PANE_TAB_STRIP_LINE, PANE_TAB_STRIP_LINE_LEFT, PANE_TAB_STRIP_LINE_RIGHT, PaneTab, PaneTabLabel } from '@/components/ui/pane-tab'
import { ContribBoundary } from '@/contrib/react/boundary'
import { useContributions } from '@/contrib/react/use-contributions'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import { $layoutEditMode } from '../../edit-mode'
import { useWindowControlsOverlap } from '../../geometry'
import type { DropPosition, GroupNode, RootEdge } from '../model'
import { adjacentGroup } from '../model'
import {
  $dropHint,
  $hiddenTreePanes,
  $layoutTree,
  $narrowViewport,
  $treeDragging,
  activateTreePane,
  closeTreePane,
  moveTreePane,
  SESSION_TILE_DRAG,
  setTreeGroupHeaderHidden,
  splitTreeZone,
  toggleTreeGroupMinimized
} from '../store'

import { type DoubleTapContext, startPaneDrag } from './drag-session'
import { paneChrome } from './track-model'

/** A directional action in the zone menu (computed per group state). */
interface ZoneMenuDirection {
  side: RootEdge
  label: string
  run: () => void
}

const DIRECTION_ORDER: readonly RootEdge[] = ['right', 'bottom', 'left', 'top']
const DIRECTION_ARROW: Record<RootEdge, string> = { bottom: '↓', left: '←', right: '→', top: '↑' }

/** Right-click zone menu: directional actions + header toggle + minimize.
 *  The directions are CONTEXTUAL (computed by TreeGroup): a stacked group
 *  offers "Split <dir>" (carve a new zone with the clicked pane — VS Code
 *  split-and-move in one gesture); a single-pane group offers "Move <dir>"
 *  into the zone actually sitting on that side — directions with no visible
 *  neighbor aren't offered, so no action ever appears to do nothing. */
function ZoneMenu({
  children,
  closable,
  minimizable = true,
  directions,
  headerHidden,
  minimized,
  nodeId
}: {
  children: ReactNode
  /** The pane the menu closes (the right-clicked chip / the active pane);
   *  undefined = not closable (the main zone). */
  closable?: () => string | undefined
  /** False for the zone hosting the uncloseable workspace — collapsing the
   *  MAIN pane strands the app behind a strip. */
  minimizable?: boolean
  directions: ZoneMenuDirection[]
  headerHidden?: boolean
  minimized?: boolean
  nodeId: string
}) {
  const { t } = useI18n()

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>{children}</ContextMenuTrigger>
      <ContextMenuContent>
        {directions.map(direction => (
          <ContextMenuItem key={direction.side} onSelect={direction.run}>
            {direction.label}
          </ContextMenuItem>
        ))}
        <ContextMenuItem onSelect={() => setTreeGroupHeaderHidden(nodeId, !headerHidden)}>
          {headerHidden ? t.zones.showHeader : t.zones.hideHeader}
        </ContextMenuItem>
        {minimizable && (
          <ContextMenuItem onSelect={() => toggleTreeGroupMinimized(nodeId, !minimized)}>
            {minimized ? t.zones.restore : t.zones.minimize}
          </ContextMenuItem>
        )}
        {/* Resolved at render: the menu mounts on open, after the right-click
            set menuPane — so an uncloseable target hides the item instead
            of offering a dead action. */}
        {closable?.() !== undefined && (
          <ContextMenuItem
            onSelect={() => {
              const paneId = closable?.()

              if (paneId) {
                closeTreePane(paneId)
              }
            }}
          >
            {t.common.close}
          </ContextMenuItem>
        )}
      </ContextMenuContent>
    </ContextMenu>
  )
}

export function TreeGroup({
  node,
  parentAxis,
  railSide = 'left'
}: {
  node: GroupNode
  parentAxis?: 'column' | 'row'
  railSide?: 'left' | 'right'
}) {
  const { t } = useI18n()
  const ref = useRef<HTMLDivElement>(null)
  const stripRef = useRef<HTMLDivElement>(null)
  // The chip under the last right-click — the pane the zone menu's Split
  // actions carry into the new zone (header background = the active pane).
  // STATE, not a ref: the menu items (incl. Close's visibility) are JSX
  // evaluated during THIS component's render — a ref write on right-click
  // doesn't re-render, so the menu showed the PREVIOUS target's items (Close
  // missing on an inactive tile tab whose zone-active was the uncloseable
  // workspace).
  const [menuPane, setMenuPane] = useState<string | undefined>(undefined)
  const panes = useContributions('panes')
  // Coarse drag flag only (set once at drag start/end). The per-frame drop
  // HINT lives in ZoneDropOverlay so a moving pointer re-renders the tiny
  // overlay, not every zone's header/body (and not the menuDirections walk).
  const dragging = useStore($treeDragging)
  const editMode = useStore($layoutEditMode)
  const wcOverlap = useWindowControlsOverlap(ref, true)

  const hiddenPanes = useStore($hiddenTreePanes)
  const narrow = useStore($narrowViewport)

  const paneFor = (id: string) => panes.find(p => p.id === id)

  // Unregistered (plugin not loaded), chrome-toggled-off, and narrow-collapsed
  // panes drop out of the header; the active pane falls back to the first
  // shown one (render-side — the tree keeps `active`).
  // Edit mode forces toggle-hidden panes visible so they can be rearranged
  // (mirrors tree-split's paneGone) — restores itself on exit.
  const paneShown = (id: string) =>
    Boolean(paneFor(id)) && (editMode || !hiddenPanes.has(id)) && !(narrow && paneChrome(paneFor(id)).collapsible)

  const shown = node.panes.filter(paneShown)
  const activeId = shown.includes(node.active) ? node.active : (shown[0] ?? node.active)
  const active = paneFor(activeId)
  const isEmpty = node.panes.length === 0

  // ONE header style: the app's compact pane-header. DEFAULT is contextual —
  // a single pane isn't a "tab", so its header auto-hides; a stack shows its
  // chips. EXCEPTION: a lone TILE (closeable, placement 'main' — a session/page
  // split) always shows its header, so it has a tab + close X — a tile in its
  // own zone was otherwise unclosable (the "3rd tile has no tab" trap). Chrome
  // panes (sessions/files/terminal…) and the uncloseable workspace keep the
  // clean no-tab default. Double-click toggles it either way; a minimized
  // group always shows its header (it IS the header).
  const hasLoneTile = shown.some(id => {
    const chrome = paneChrome(paneFor(id))

    return !chrome.uncloseable && chrome.placement === 'main'
  })

  // A full-page view (headerVeto) suppresses the strip while it's the active
  // pane — a page is not a tab-able surface; the bar returns with the chat.
  const headerHidden = paneChrome(active).headerVeto || (node.headerHidden ?? (shown.length <= 1 && !hasLoneTile))

  // A group collapses ALONG its parent split's axis. In a row that means the
  // WIDTH collapses — a full-width horizontal header would strand a tall
  // empty column, so the minimized form is a narrow vertical rail instead
  // (tabs reading top-to-bottom). In a column (stacked zones) the horizontal
  // header IS the collapsed form, exactly as before.
  const verticalCollapse = Boolean(node.minimized) && parentAxis === 'row' && !isEmpty
  const headerVisible = !isEmpty && !verticalCollapse && (Boolean(node.minimized) || !headerHidden)

  // Drag handles preventDefault pointerdown (no native dblclick), so the
  // header + chips share a synthesized double-tap: restore if collapsed
  // (undoing the first tap's minimize toggle) and hide the chrome.
  const hideHeaderDoubleTap: DoubleTapContext = {
    key: `hide-header-${node.id}`,
    onDoubleTap: () => {
      toggleTreeGroupMinimized(node.id, false)
      setTreeGroupHeaderHidden(node.id, true)
    }
  }

  const dirWord: Record<RootEdge, string> = {
    bottom: t.zones.dirDown,
    left: t.zones.dirLeft,
    right: t.zones.dirRight,
    top: t.zones.dirUp
  }

  // Zone-menu directions, contextual to this group's state:
  //  - stacked panes -> "Split <dir>": carve a new zone on that side with the
  //    right-clicked chip's pane in it (split + move, one gesture);
  //  - a single pane -> "Move <dir>": join the zone visually adjacent on that
  //    side (splitting here would only make an invisible empty zone). Sides
  //    with no visible neighbor are omitted entirely.
  const tree = useStore($layoutTree)

  const menuDirections: ZoneMenuDirection[] =
    shown.length > 1
      ? DIRECTION_ORDER.map(side => ({
          side,
          label: `${t.zones.split(dirWord[side])} ${DIRECTION_ARROW[side]}`,
          run: () => splitTreeZone(node.id, side, menuPane ?? activeId)
        }))
      : DIRECTION_ORDER.flatMap(side => {
          const neighbor = tree ? adjacentGroup(tree, node.id, side, g => g.panes.some(paneShown)) : null

          if (!neighbor || neighbor.id === node.id) {
            return []
          }

          return [
            {
              side,
              label: `${t.zones.move(dirWord[side])} ${DIRECTION_ARROW[side]}`,
              run: () => moveTreePane(activeId, { groupId: neighbor.id, pos: 'center' })
            }
          ]
        })

  // Close targets the right-clicked chip (falling back to the active pane);
  // only panes that declare `uncloseable` (the main workspace) are exempt.
  const closable = () => {
    const paneId = menuPane ?? activeId

    return paneChrome(paneFor(paneId)).uncloseable ? undefined : paneId
  }

  // The zone hosting the uncloseable workspace never minimizes — collapsing
  // MAIN strands the whole app behind a strip.
  const minimizable = !shown.some(id => paneChrome(paneFor(id)).uncloseable)

  // Same menu on the header strip and the edit veil — one prop bag.
  const zoneMenu = {
    closable,
    directions: menuDirections,
    headerHidden,
    minimizable,
    minimized: node.minimized,
    nodeId: node.id
  }

  // NO body double-click toggle: virtualized content (the thread) recreates
  // its nodes between clicks, so the gesture was hopelessly unreliable. The
  // bar's lifecycle is explicit instead — gaining a tab sticky-shows it
  // (insertAtGroup pins headerHidden false), the main tab's context menu
  // hides it, and full-page views veto it via paneChrome.headerVeto.

  return (
    <div
      className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-(--ui-bg-editor)"
      data-tree-group={node.id}
      // Advertises the visible tab strip so panes can drop their own
      // self-naming labels (see [data-pane-self-label] in styles.css).
      data-zone-header={headerVisible || undefined}
      ref={ref}
      style={wcOverlap ? { paddingTop: wcOverlap.y + wcOverlap.height } : undefined}
    >
      {wcOverlap && (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute z-10 [-webkit-app-region:drag]"
          style={{ height: wcOverlap.height, left: wcOverlap.x, top: wcOverlap.y, width: wcOverlap.width }}
        />
      )}

      {/* Minimized in a ROW: a narrow vertical rail — same PaneTab shell as
          the horizontal strip, just `vertical`. Click a tab to restore +
          activate; click anywhere else on the rail to restore. */}
      {verticalCollapse && (
        <ZoneMenu {...zoneMenu}>
          <div
            className={cn(
              'flex h-full w-7 shrink-0 cursor-pointer select-none flex-col items-stretch bg-(--pane-tab-strip-bg) [--pane-tab-strip-bg:var(--theme-card-seed)]',
              // Strip line faces the content the zone collapsed away from.
              railSide === 'right' ? PANE_TAB_STRIP_LINE_LEFT : PANE_TAB_STRIP_LINE_RIGHT
            )}
            onClick={() => toggleTreeGroupMinimized(node.id, false)}
            title={t.zones.restore}
          >
            <div
              className="flex min-h-0 flex-col overflow-y-auto [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
              role="tablist"
            >
              {shown.map(paneId => {
                const closeable = !paneChrome(paneFor(paneId)).uncloseable
                const title = paneFor(paneId)?.title ?? paneId

                return (
                  <PaneTab
                    // Match the horizontal minimized strip: no tab is "active"
                    // while collapsed (there's no content surface to merge into).
                    aria-selected={paneId === activeId}
                    data-tree-tab={paneId}
                    key={paneId}
                    onClick={event => {
                      event.stopPropagation()
                      toggleTreeGroupMinimized(node.id, false)
                      activateTreePane(node.id, paneId)
                    }}
                    onClose={closeable ? () => closeTreePane(paneId) : undefined}
                    role="tab"
                    side={railSide}
                    vertical
                  >
                    <PaneTabLabel>{title}</PaneTabLabel>
                  </PaneTab>
                )
              })}
            </div>
          </div>
        </ZoneMenu>
      )}

      {/* Header: the file-preview tab strip (PaneTab), one shared component. */}
      {headerVisible && (
        <ZoneMenu {...zoneMenu}>
          <div
            // Active = sidebar surface (merges into body). Strip =
            // `--theme-card-seed` (VS Code `tab.inactiveBackground`). Line =
            // PANE_TAB_STRIP_LINE; active tab cuts through it.
            // data-zone-tabstrip: a drop over here STACKS (drag-session reads it).
            className={cn(
              'group/pane-header relative flex h-7 shrink-0 select-none bg-(--pane-tab-strip-bg) [-webkit-app-region:no-drag] [--pane-tab-active-bg:var(--ui-sidebar-surface-background)] [--pane-tab-strip-bg:var(--theme-card-seed)]',
              PANE_TAB_STRIP_LINE
            )}
            data-zone-tabstrip={node.id}
            onContextMenu={e => {
              setMenuPane((e.target as HTMLElement).closest('[data-tree-tab]')?.getAttribute('data-tree-tab') ?? undefined)
            }}
            onPointerDown={e =>
              // Tap the header to collapse to it / expand back — the DetailPane
              // / sidebar-section gesture (never for the main zone). Double-tap
              // hides the header entirely. Drag still moves the pane.
              startPaneDrag(
                activeId,
                e,
                () => minimizable && toggleTreeGroupMinimized(node.id, !node.minimized),
                undefined,
                hideHeaderDoubleTap
              )
            }
            ref={stripRef}
            style={{ cursor: 'grab' }}
          >
            <div
              className="flex min-w-0 flex-1 overflow-x-auto overflow-y-hidden [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
              role="tablist"
            >
              {shown.map(paneId => {
                const isActive = paneId === activeId && !node.minimized
                const chrome = paneChrome(paneFor(paneId))
                const closeable = !chrome.uncloseable
                const title = paneFor(paneId)?.title ?? paneId

                const tab = (
                  <PaneTab
                    active={isActive}
                    aria-selected={isActive}
                    data-tree-tab={paneId}
                    key={paneId}
                    onClose={closeable ? () => closeTreePane(paneId) : undefined}
                    onPointerDown={e =>
                      startPaneDrag(
                        paneId,
                        e,
                        () => {
                          // Tabs ACTIVATE (restoring a collapsed group).
                          // Minimize lives on the chevron / single-pane label
                          // — overloading the active tab made double-click a
                          // minimize/restore/hide lottery.
                          if (node.minimized) {
                            toggleTreeGroupMinimized(node.id, false)
                          }

                          activateTreePane(node.id, paneId)
                        },
                        stripRef.current ? { groupId: node.id, strip: stripRef.current } : undefined,
                        hideHeaderDoubleTap
                      )
                    }
                    role="tab"
                    style={{ cursor: 'grab' }}
                  >
                    <PaneTabLabel>{title}</PaneTabLabel>
                  </PaneTab>
                )

                // A pane may wrap ITS tab in a domain menu (session verbs on a
                // tile tab); the wrapper needs the key since it's the root.
                return <Fragment key={paneId}>{chrome.tabWrap ? chrome.tabWrap(tab) : tab}</Fragment>
              })}
            </div>
            {minimizable && (
              <button
                aria-label={node.minimized ? t.zones.restore : t.zones.minimize}
                className="mx-1 grid size-5 shrink-0 place-items-center self-center rounded-md text-(--ui-text-tertiary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground focus-visible:opacity-100 group-hover/pane-header:opacity-100"
                onClick={() => toggleTreeGroupMinimized(node.id, !node.minimized)}
                onPointerDown={e => e.stopPropagation()}
                type="button"
              >
                <Codicon name={node.minimized ? 'chevron-down' : 'chevron-up'} size="0.75rem" />
              </button>
            )}
            <StripDropCaret groupId={node.id} stripRef={stripRef} />
          </div>
        </ZoneMenu>
      )}

      {/* Body: the active pane's contributed content, or the empty zone. */}
      {!node.minimized && (
        <div className="relative min-h-0 min-w-0 flex-1 overflow-auto">
          {isEmpty ? (
            <div className="grid h-full place-items-center">
              {/* Same decode primitive as the CONNECTING boot overlay. */}
              <DecodeText className="text-(--ui-text-quaternary)" cursor prefix={1} text="HERMES" />
            </div>
          ) : active?.render ? (
            <ContribBoundary id={active.id}>{active.render()}</ContribBoundary>
          ) : (
            <div className="p-3 font-mono text-[11px] text-(--ui-text-quaternary)">{t.zones.missingPane(activeId)}</div>
          )}
        </div>
      )}

      {/* Edit-mode veil: the BODY is a drag handle for the active pane. It
          starts below the header so tabs/headers stay directly interactive
          (drag any tab, right-click for the zone menu). */}
      {editMode && !dragging && !isEmpty && !node.minimized && (
        <ZoneMenu {...zoneMenu}>
          <div
            // z-50: pane CONTENT may carry its own stacked chrome (the
            // terminal rail is z-40) — the edit veil must cover all of it.
            // The scrim mixes the accent over the CHROME BG (not transparent)
            // so it properly dims content in dark themes instead of leaving a
            // barely-tinted wash; the light blur reads as "edit mode" the same
            // way the zone editor's backdrop does.
            className="absolute inset-x-0 bottom-0 z-50 flex cursor-grab items-center justify-center outline-1 -outline-offset-2 outline-dashed backdrop-blur-[2px]"
            onPointerDown={e => startPaneDrag(activeId, e)}
            style={{
              top: headerVisible ? 28 : 0,
              background:
                'color-mix(in srgb, var(--ui-accent) 6%, color-mix(in srgb, var(--ui-bg-chrome) 55%, transparent))',
              outlineColor: 'color-mix(in srgb, var(--ui-accent) 55%, transparent)'
            }}
          >
            <span className="flex max-w-[calc(100%-1rem)] items-center gap-1.5 rounded-md border border-(--ui-stroke-secondary) bg-popover px-2 py-1 text-[0.64rem] font-semibold uppercase tracking-[0.16em] text-(--ui-text-secondary)">
              <Codicon className="shrink-0" name="gripper" size="0.8125rem" />
              <span className="min-w-0 truncate">{active?.title ?? activeId}</span>
            </span>
          </div>
        </ZoneMenu>
      )}

      {/* FancyZones drop overlay — its own component so the per-frame drop
          hint re-renders only this (tiny) node, not the whole zone. */}
      <ZoneDropOverlay isEmpty={isEmpty} node={node} />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab-strip insertion caret
// ---------------------------------------------------------------------------

/**
 * The insertion divider for a stack drop: a 2px vertical line at the slot the
 * dragged tab will land in (before `stack.before`, or after the last tab).
 * Absolute over the strip — pure overlay, zero layout shift. #000 on light,
 * #FFF on dark. Split out so per-pointermove `$dropHint` churn re-renders
 * only this node (same isolation contract as ZoneDropOverlay).
 */
function StripDropCaret({ groupId, stripRef }: { groupId: string; stripRef: RefObject<HTMLDivElement | null> }) {
  const hint = useStore($dropHint)
  const strip = stripRef.current
  const stack = hint?.groupId === groupId ? hint.stack : undefined

  if (stack === undefined || !strip) {
    return null
  }

  // Slot x: the before-tab's left edge, or the last tab's right edge.
  const tabs = [...strip.querySelectorAll<HTMLElement>('[data-tree-tab]')]
  const target = stack.before ? tabs.find(el => el.dataset.treeTab === stack.before) : tabs.at(-1)

  if (!target) {
    return null
  }

  const stripRect = strip.getBoundingClientRect()
  const targetRect = target.getBoundingClientRect()
  const x = (stack.before ? targetRect.left : targetRect.right) - stripRect.left

  // A short centered tick (~60% of the tab), not a full-height wall — reads
  // as an insertion point between labels, browser-tab style.
  return (
    <span
      aria-hidden
      className="pointer-events-none absolute z-50 w-px -translate-x-1/2 bg-black dark:bg-white"
      style={{
        height: targetRect.height * 0.6,
        left: x,
        top: targetRect.top - stripRect.top + targetRect.height * 0.2
      }}
    />
  )
}

// ---------------------------------------------------------------------------
// FancyZones drop overlay
// ---------------------------------------------------------------------------

/** Overlay entry fade. FancyZones ships 200ms (FADE_IN_DURATION_MILLIS in
 *  zones-engine); on a drag that starts under the cursor that ramp reads as
 *  lag, so the sheets snap in far faster — same softening, instant feel. */
const OVERLAY_FADE_MS = 80

/** Sheet inset from the zone edge (px). */
const REGION_PAD = 6

/** The sheet's box per drop position — longhand insets so CSS transitions can
 *  interpolate the px↔% change: the target GLIDES between the full zone and
 *  the hovered half instead of snapping (VS Code dock preview). */
const REGION: Record<DropPosition, CSSProperties> = {
  bottom: { bottom: REGION_PAD, left: REGION_PAD, right: REGION_PAD, top: '50%' },
  center: { bottom: REGION_PAD, left: REGION_PAD, right: REGION_PAD, top: REGION_PAD },
  left: { bottom: REGION_PAD, left: REGION_PAD, right: '50%', top: REGION_PAD },
  right: { bottom: REGION_PAD, left: '50%', right: REGION_PAD, top: REGION_PAD },
  top: { bottom: '50%', left: REGION_PAD, right: REGION_PAD, top: REGION_PAD }
}

const EDGE_ARROW: Record<Exclude<DropPosition, 'center'>, string> = {
  bottom: 'arrow-down',
  left: 'arrow-left',
  right: 'arrow-right',
  top: 'arrow-up'
}

/**
 * The FancyZones drop overlay for one zone. Split out of TreeGroup so the
 * per-pointermove `$dropHint` churn re-renders only this lightweight node —
 * the zone's header, body, and menu-direction walk stay put during a drag.
 *
 * ONE dashed sheet per zone, in the attachment dropzone's design language
 * (DROP_SHEET_CLASS + DropPill — the composer drop and the zone targets speak
 * identically): a quiet outline over every eligible zone, accent-lit over the
 * target, morphing to the hovered half for an edge split. The pill names the
 * outcome; edges get their arrow.
 */
function ZoneDropOverlay({ isEmpty, node }: { isEmpty: boolean; node: GroupNode }) {
  const { t } = useI18n()
  const dragging = useStore($treeDragging)
  const hint = useStore($dropHint)

  if (dragging === null) {
    return null
  }

  // A session drag (sidebar row) reuses this exact overlay — over ANY zone
  // now (stack into its tabs / split its edges); only a CHAT zone's center is
  // a link-to-chat (the composer overlay owns that visual).
  const sessionDrag = dragging === SESSION_TILE_DRAG
  const chatZone = node.panes.some(p => p === 'workspace' || p.startsWith('session-tile:'))

  const isDragSource = node.panes.includes(dragging)

  // The source zone, when it holds only the dragged pane, has nothing to drop.
  if (isDragSource && node.panes.length === 1) {
    return null
  }

  const primary = hint?.groupId === node.id

  // Hovering the target's TAB STRIP: the insertion caret (StripDropCaret)
  // owns the affordance — the zone sheet stands down so the two never stack.
  if (primary && hint?.stack !== undefined) {
    return null
  }

  const active = hint?.groupIds?.includes(node.id) ?? false
  const multi = (hint?.groupIds?.length ?? 0) > 1
  // Sub-positions only exist for a single-zone target (a Shift-span merges).
  const pos = primary && !multi ? (hint?.pos ?? 'center') : 'center'
  // Session drag over a CHAT zone's CENTER: the "link to chat" overlay inside
  // the surface (ChatDropOverlay — the same sheet + pill) owns that region;
  // this sheet fades out so the two never stack. A non-chat zone's center has
  // no chat to link, so it shows the normal stack sheet. Edges act like a tab.
  const centerLink = sessionDrag && primary && pos === 'center' && chatZone

  const pill =
    !primary || centerLink
      ? null
      : multi
        ? { icon: 'combine', label: t.zones.spanHere }
        : pos !== 'center'
          ? { icon: EDGE_ARROW[pos], label: sessionDrag ? t.zones.openHere : t.zones.splitHere }
          : isDragSource
            ? { icon: 'discard', label: t.zones.staysHere }
            : { icon: 'layers', label: isEmpty ? t.zones.moveHere : t.zones.stackHere }

  return (
    <div
      className="pointer-events-none absolute inset-0 z-40"
      style={{ animation: `hermes-zone-fade ${OVERLAY_FADE_MS}ms linear both` }}
    >
      <div
        className={cn(
          DROP_SHEET_CLASS,
          // Transition ONLY the box + colors. `transition-all` also animated
          // backdrop-filter, and a blur interpolating while the insets glide
          // re-blurs half a zone every frame — the single most expensive
          // paint in the whole drag.
          'absolute flex items-center justify-center transition-[top,right,bottom,left,background-color,border-color,opacity] duration-150 ease-out',
          // Blur only the live target — idle outlines must not fog the app.
          active && !centerLink && DROP_SHEET_BLUR_CLASS,
          centerLink && 'opacity-0'
        )}
        style={{
          ...REGION[pos],
          // Accent over a card wash so the fill dims content on dark themes
          // (a bare accent alpha disappears there).
          background: active
            ? 'color-mix(in srgb, var(--ui-accent) 18%, color-mix(in srgb, var(--dt-card) 55%, transparent))'
            : 'color-mix(in srgb, var(--ui-accent) 5%, color-mix(in srgb, var(--dt-card) 25%, transparent))',
          borderColor: `color-mix(in srgb, var(--ui-accent) ${active ? 75 : 28}%, transparent)`
        }}
      >
        {pill && <DropPill icon={pill.icon}>{pill.label}</DropPill>}
      </div>
    </div>
  )
}
