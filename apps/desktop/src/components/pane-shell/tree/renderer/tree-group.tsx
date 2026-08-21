/**
 * Group node renderer — a ZONE: header strip (tabs when stacked, minimize
 * chevron) + the active pane's content, resolved from the contribution
 * registry (`area: 'panes'`). Empty zones exist only in editor-authored
 * trees (drop targets until the first structural op prunes them).
 *
 * Dragging is FancyZones-style (drag-session.ts): the layout stays fixed and
 * every zone lights up as a whole-region drop target. Right-click opens the
 * contextual zone menu (tab close verbs + header/minimize toggles).
 */

import { useStore } from '@nanostores/react'
import { type CSSProperties, Fragment, type ReactNode, type RefObject, useRef, useState } from 'react'

import { ActionsContextMenu, type MenuKit, renderActionItem } from '@/components/ui/actions-menu'
import { Codicon } from '@/components/ui/codicon'
import { DecodeText } from '@/components/ui/decode-text'
import { DROP_SHEET_BLUR_CLASS, DROP_SHEET_CLASS } from '@/components/ui/drop-affordance'
import {
  PANE_TAB_STRIP_LINE_LEFT,
  PANE_TAB_STRIP_LINE_RIGHT,
  PaneStripGlyph,
  PaneTab,
  paneTabCloseItems,
  PaneTabLabel,
  PaneTabStrip
} from '@/components/ui/pane-tab'
import { ContribBoundary, ContribRender } from '@/contrib/react/boundary'
import { useContributions } from '@/contrib/react/use-contributions'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import { $layoutEditMode } from '../../edit-mode'
import { useWindowControlsOverlap } from '../../geometry'
import { emptyPaneLifecycleState, reconcilePaneLifecycle } from '../../pane-lifecycle'
import { hiddenPaneProps, PaneGroupContext, PaneLifecycleContext, PaneVisibleContext } from '../../pane-visibility'
import type { DropPosition, GroupNode } from '../model'
import {
  $dropHint,
  $hiddenTreePanes,
  $narrowViewport,
  $newSessionTabAction,
  $panesWithCloser,
  $treeDragging,
  $treePaneEpochs,
  activateTreePane,
  closeAllTreeTabs,
  closeOtherTreeTabs,
  closeTabPane,
  closeTreeTabsToRight,
  collapseTreePane,
  hideOnlyZoneTabs,
  isCollapsePane,
  isMainStripPane,
  isSessionStripPane,
  noteActiveTreeGroup,
  reloadTreePane,
  restoreTreePane,
  SESSION_TILE_DRAG,
  setStripTabHidden,
  setTreeGroupHeaderHidden,
  setTreeGroupMinimized,
  treeTabCloseTargets
} from '../store'
import {
  $tabSelection,
  clearTabSelection,
  isToggleSelectClick,
  selectionFor,
  selectTabRange,
  toggleTabSelected
} from '../tab-selection'

import { type DoubleTapContext, startPaneDrag } from './drag-session'
import { forceLoneHeaderForPanes } from './lone-header'
import { useActiveTabVisible } from './tab-strip-scroll'
import { paneChrome } from './track-model'

/** Right-click zone menu: the tab verbs (close this / others / to the right /
 *  all) plus the strip's own chrome toggles. Same items and icons as a session
 *  tab's menu, so every tab in a strip answers a right-click the same way —
 *  a pane with no domain menu of its own (the file tree, a terminal, the main
 *  tab on a fresh draft) falls through to this one. */
function ZoneMenu({
  children,
  closable,
  minimizable = true,
  headerHidden,
  minimized,
  nodeId,
  targetPane
}: {
  children: ReactNode
  /** The pane the menu closes (the right-clicked chip / the active pane);
   *  undefined = not closable (the main zone). */
  closable?: () => string | undefined
  /** False for the zone hosting the uncloseable workspace — collapsing the
   *  MAIN pane strands the app behind a strip. */
  minimizable?: boolean
  headerHidden?: boolean
  minimized?: boolean
  nodeId: string
  /** The right-clicked chip (else the active pane) — what the close-others /
   *  to-the-right / all verbs measure from. Called when the menu RENDERS, not
   *  on every zone re-render: resolving the siblings reads the layout tree,
   *  and subscribing every zone to it made a sash drag re-render every
   *  mounted pane. */
  targetPane: () => string
}) {
  const { t } = useI18n()

  // Resolved at render: the menu mounts on open, after the right-click set
  // menuPane — so an uncloseable target hides Close instead of offering a
  // dead action, and the counts describe the chip actually clicked.
  const items = (kit: MenuKit) => {
    const paneId = closable?.()
    const targetId = targetPane()

    return (
      <>
        {renderActionItem(kit, {
          icon: 'refresh',
          label: t.zones.reload,
          onSelect: () => reloadTreePane(targetId)
        })}
        <kit.Separator />
        {paneTabCloseItems(kit, {
          counts: treeTabCloseTargets(targetId),
          onClose: paneId !== undefined ? () => closeTabPane(paneId) : undefined,
          onCloseAll: () => closeAllTreeTabs(targetId),
          onCloseOthers: () => closeOtherTreeTabs(targetId),
          onCloseToRight: () => closeTreeTabsToRight(targetId)
        })}
        {(() => {
          // Show/hide rows for the zone's hide-only chrome tabs (sessions /
          // Bots) — their Close replacement. Resolved when the menu OPENS,
          // same no-subscription contract as the close-verb counts above.
          const hideOnly = hideOnlyZoneTabs(nodeId)

          if (hideOnly.length === 0) {
            return null
          }

          return (
            <>
              <kit.Separator />
              {hideOnly.map(tab =>
                renderActionItem(kit, {
                  icon: tab.hidden ? 'eye' : 'eye-closed',
                  key: `strip-tab-${tab.id}`,
                  label: tab.hidden ? t.zones.showStripTab(tab.title) : t.zones.hideStripTab(tab.title),
                  onSelect: () => setStripTabHidden(tab.id, !tab.hidden)
                })
              )}
            </>
          )
        })()}
        <kit.Separator />
        {renderActionItem(kit, {
          icon: headerHidden ? 'eye' : 'eye-closed',
          label: headerHidden ? t.zones.showHeader : t.zones.hideHeader,
          onSelect: () => setTreeGroupHeaderHidden(nodeId, !headerHidden)
        })}
        {minimizable &&
          renderActionItem(kit, {
            // Same action-direction contract as the strip button below: the
            // icon points where the zone will GO (restore opens upward).
            icon: minimized ? 'chevron-up' : 'chevron-down',
            label: minimized ? t.zones.restore : t.zones.minimize,
            onSelect: () => setTreeGroupMinimized(nodeId, !minimized)
          })}
      </>
    )
  }

  return (
    <ActionsContextMenu contentClassName="w-40" items={items}>
      {children}
    </ActionsContextMenu>
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
  // The scrolling tab list inside the header (the strip also holds the
  // minimize chevron, which must not scroll away).
  const tabsRef = useRef<HTMLDivElement>(null)
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
  const newSessionTabAction = useStore($newSessionTabAction)
  const panesWithCloser = useStore($panesWithCloser)
  // Multi-tab selection (⌥/Ctrl-click, Shift-click) — null for every zone but
  // the one holding it, so this subscription is quiet during normal use.
  const tabSelection = useStore($tabSelection)
  // Reload epochs: only an explicit tab-menu Reload writes here, so this
  // subscription costs nothing on a normal render.
  const paneEpochs = useStore($treePaneEpochs)

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

  // BOUNDED KEEP-ALIVE: the active pane is visible, a small per-zone LRU stays
  // hot-hidden, and older panes park (unmount). This preserves fast tab
  // round-trips without letting a long-lived zone pin every transcript it has
  // ever visited. Stateful resources can opt out of parking (the terminal keeps
  // its PTY alive while hidden). Lazy remains deliberate: restored background
  // tabs have no lifecycle entry and do not mount until first activation.
  const lifecycleRef = useRef(emptyPaneLifecycleState())

  if (!node.minimized && !isEmpty) {
    lifecycleRef.current = reconcilePaneLifecycle(lifecycleRef.current, {
      activeId,
      keepAlive: id => Boolean(paneChrome(paneFor(id)).lifecycleKeepAlive),
      paneIds: shown
    })
  }

  const paneLifecycle = lifecycleRef.current.entries
  const keptPanes = shown.filter(id => paneLifecycle[id] && paneLifecycle[id].lifecycle !== 'parked')

  // ONE header style: the app's compact pane-header. DEFAULT is contextual —
  // a single pane isn't a "tab", so its header auto-hides; a stack shows its
  // chips. EXCEPTIONS force a lone pane to keep its header (tab + close X):
  //  - a TILE (closeable, placement 'main' — a session/page split), else a
  //    tile in its own zone is unclosable (the "3rd tile has no tab" trap);
  //  - a TOOL PANEL (terminal/logs — a collapse pane) dragged out of the main
  //    stack, else it's a dead zone with no tab to grab or ✕ to close.
  // The uncloseable workspace and side chrome (sessions/files) keep the clean
  // no-tab default. Double-click toggles it either way; a minimized group
  // always shows its header (it IS the header).
  // Session-tile ids force the header even before chrome registers — cycling
  // onto a freshly-split tile used to land headerless ("name card missing").
  const forceLoneHeader = forceLoneHeaderForPanes(shown, id => paneChrome(paneFor(id)), isCollapsePane)

  // A full-page view (headerVeto) suppresses the strip while it's the active
  // pane — a page is not a tab-able surface; the bar returns with the chat.
  const headerHidden = paneChrome(active).headerVeto || (node.headerHidden ?? (shown.length <= 1 && !forceLoneHeader))

  // A group collapses ALONG its parent split's axis. In a row that means the
  // WIDTH collapses — a full-width horizontal header would strand a tall
  // empty column, so the minimized form is a narrow vertical rail instead
  // (tabs reading top-to-bottom). In a column (stacked zones) the horizontal
  // header IS the collapsed form, exactly as before.
  const verticalCollapse = Boolean(node.minimized) && parentAxis === 'row' && !isEmpty
  const headerVisible = !isEmpty && !verticalCollapse && (Boolean(node.minimized) || !headerHidden)

  // Keep the activated tab — and, on the last one, the trailing "+" — inside
  // the strip's scroll window. Opening a tab past the right edge otherwise
  // left both the new tab and the button that made it out of view.
  useActiveTabVisible(tabsRef, activeId, {
    enabled: headerVisible,
    last: shown[shown.length - 1] === activeId,
    tabCount: shown.length
  })

  // Drag handles preventDefault pointerdown (no native dblclick), so the
  // header + chips share a synthesized double-tap: restore if collapsed
  // (undoing the first tap's minimize toggle) and hide the chrome.
  const hideHeaderDoubleTap: DoubleTapContext = {
    key: `hide-header-${node.id}`,
    onDoubleTap: () => {
      setTreeGroupMinimized(node.id, false)
      setTreeGroupHeaderHidden(node.id, true)
    }
  }

  // Zone-menu close targets read the layout tree, but this component must NOT
  // subscribe to it: `useStore($layoutTree)` here wires every zone — and
  // therefore every mounted pane and its whole transcript — to the entire
  // tree. A sash drag rewrites the tree once per frame, so dragging the
  // sidebar re-rendered all five tiles' message lists on every pointermove
  // (measured: TreeGroup 180 renders cascading into ChatView/Thread/TileChat
  // at ~4.5s each, holding the drag at ~3fps). The menu's items are resolved
  // when it OPENS, so they read the tree with `.get()` at that moment instead.
  const targetPane = () => menuPane ?? activeId

  // Close targets the right-clicked chip (falling back to the active pane);
  // panes that declare `uncloseable` (the main workspace) or `hideOnly`
  // (sessions / Bots — show/hide replaces Close) are exempt.
  const closable = () => {
    const paneId = targetPane()
    const chrome = paneChrome(paneFor(paneId))

    return chrome.uncloseable || chrome.hideOnly ? undefined : paneId
  }

  // The zone hosting the uncloseable workspace never minimizes — collapsing
  // MAIN strands the whole app behind a strip.
  const minimizable = !shown.some(id => paneChrome(paneFor(id)).uncloseable)

  // Middle-click / ⌘-click on a tab: one routing for every tab kind, the same
  // one the zone menu's Close and ⌘W use.
  const closeTab = (paneId: string) => closeTabPane(paneId)

  // A pane whose store owns Close keeps the gesture even when the pane itself
  // is uncloseable — the workspace tab empties to a fresh draft rather than
  // leaving the tree. Hide-only chrome (sessions / Bots) opts out of every
  // close gesture: its tabs are shown/hidden (zone menu, ⌘K), never closed —
  // an accidental ✕ on standing chrome removed Bot Mode until the next launch.
  const closeableTab = (paneId: string) =>
    !paneChrome(paneFor(paneId)).hideOnly && (!paneChrome(paneFor(paneId)).uncloseable || panesWithCloser.has(paneId))

  // A pane's own live label when it has one, else its registered string.
  const tabLabel = (paneId: string) => paneChrome(paneFor(paneId)).tabTitle?.() ?? paneFor(paneId)?.title ?? paneId

  // Collapse/restore a tool panel (or plain minimize elsewhere) — the header
  // chevron + tap gesture, routed so ⌃`/the titlebar toggle stay truthful.
  const toggleCollapse = () => (node.minimized ? restoreTreePane(activeId) : collapseTreePane(activeId))

  // Same menu on the header strip and the edit veil — one prop bag.
  const zoneMenu = {
    closable,
    headerHidden,
    minimizable,
    minimized: node.minimized,
    nodeId: node.id,
    targetPane
  }

  // NO body double-click toggle: virtualized content (the thread) recreates
  // its nodes between clicks, so the gesture was hopelessly unreliable. The
  // bar's lifecycle is explicit instead — gaining a tab sticky-shows it
  // (insertAtGroup pins headerHidden false), the main tab's context menu
  // hides it, and full-page views veto it via paneChrome.headerVeto.

  return (
    <div
      className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-(--ui-editor-surface-background)"
      data-tree-group={node.id}
      // Advertises the visible tab strip so panes can drop their own
      // self-naming labels (see [data-pane-self-label] in styles.css).
      data-zone-header={headerVisible || undefined}
      // The zone menu opens from the strip, the rail, the edit veil and the
      // body. Only the strip can name a chip, so resolve the target HERE for
      // every one of them — otherwise a right-click off the strip reused the
      // PREVIOUS target, and landing on the uncloseable workspace dropped
      // Close from the menu for a pane that closes fine.
      onContextMenu={e => {
        setMenuPane((e.target as HTMLElement).closest('[data-tree-tab]')?.getAttribute('data-tree-tab') ?? undefined)
      }}
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
              'flex h-full w-7 shrink-0 cursor-pointer select-none flex-col items-stretch bg-(--ui-sidebar-surface-background)',
              // Strip line faces the content the zone collapsed away from.
              railSide === 'right' ? PANE_TAB_STRIP_LINE_LEFT : PANE_TAB_STRIP_LINE_RIGHT
            )}
            onClick={() => restoreTreePane(activeId)}
            title={t.zones.restore}
          >
            <div
              className="flex min-h-0 flex-col overflow-y-auto [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
              role="tablist"
            >
              {shown.map(paneId => {
                const closeable = closeableTab(paneId)

                return (
                  <PaneTab
                    // Match the horizontal minimized strip: no tab is "active"
                    // while collapsed (there's no content surface to merge into).
                    aria-selected={paneId === activeId}
                    data-tree-tab={paneId}
                    key={paneId}
                    onClick={event => {
                      event.stopPropagation()
                      restoreTreePane(paneId)
                    }}
                    onClose={closeable ? () => closeTab(paneId) : undefined}
                    role="tab"
                    side={railSide}
                    vertical
                  >
                    <PaneTabLabel>{tabLabel(paneId)}</PaneTabLabel>
                  </PaneTab>
                )
              })}
            </div>
          </div>
        </ZoneMenu>
      )}

      {/* Header: the shared pane tab strip (PaneTabStrip + PaneTab). */}
      {headerVisible && (
        <ZoneMenu {...zoneMenu}>
          <PaneTabStrip
            // data-zone-tabstrip: a drop over here STACKS (drag-session reads it).
            data-zone-tabstrip={node.id}
            listRef={tabsRef}
            onPointerDown={e =>
              // Tap the header to collapse to it / expand back — the DetailPane
              // / sidebar-section gesture (never for the main zone). Double-tap
              // hides the header entirely. Drag still moves the pane.
              startPaneDrag(
                activeId,
                e,
                () => minimizable && toggleCollapse(),
                undefined,
                hideHeaderDoubleTap,
                active?.title ?? activeId
              )
            }
            ref={stripRef}
            style={{ cursor: 'grab' }}
            trailing={
              <>
                {minimizable && (
                  <button
                    aria-label={node.minimized ? t.zones.restore : t.zones.minimize}
                    className="mx-1 grid size-5 shrink-0 place-items-center self-center rounded-md text-(--ui-text-tertiary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground focus-visible:opacity-100 group-hover/pane-header:opacity-100"
                    onClick={toggleCollapse}
                    onPointerDown={e => e.stopPropagation()}
                    type="button"
                  >
                    <Codicon name={node.minimized ? 'chevron-up' : 'chevron-down'} size="0.75rem" />
                  </button>
                )}
                <StripDropCaret groupId={node.id} stripRef={stripRef} />
              </>
            }
          >
            {shown.map(paneId => {
              const isActive = paneId === activeId && !node.minimized
              const chrome = paneChrome(paneFor(paneId))
              const closeable = closeableTab(paneId)
              const title = paneFor(paneId)?.title ?? paneId
              const isSelected = tabSelection?.groupId === node.id && tabSelection.ids.has(paneId)

              const tab = (
                <PaneTab
                  active={isActive}
                  aria-selected={isActive}
                  data-tree-tab={paneId}
                  key={paneId}
                  onClose={closeable ? () => closeTab(paneId) : undefined}
                  onPointerDown={e => {
                    // Chrome's tab-selection grammar, ahead of activate/drag:
                    // Shift-click ranges from the anchor, ⌥-click (Ctrl-click
                    // off-Mac) toggles. Neither activates nor starts a drag —
                    // the press IS the selection edit. ⌘-click stays close
                    // (PaneTab claims it first) and ⌃-click stays the macOS
                    // context menu.
                    if (e.button === 0 && e.shiftKey) {
                      e.preventDefault()
                      e.stopPropagation()
                      selectTabRange(node.id, shown, paneId, activeId)

                      return
                    }

                    if (isToggleSelectClick(e)) {
                      e.preventDefault()
                      e.stopPropagation()
                      toggleTabSelected(node.id, paneId, activeId)

                      return
                    }

                    // Tabs ACTIVATE (restoring a collapsed group). Minimize
                    // lives on the chevron / single-pane label — overloading
                    // the active tab made double-click a minimize/restore/hide
                    // lottery. A plain click also collapses any multi-tab
                    // selection back to the one tab (Chrome semantics).
                    const onTap = () => {
                      clearTabSelection()

                      if (node.minimized) {
                        restoreTreePane(paneId)
                      }

                      activateTreePane(node.id, paneId)
                    }

                    // Claim the press so the STRIP's own pane-drag handler
                    // (parent onPointerDown) can't also fire. startPaneDrag
                    // does this internally; the session drag (shared with
                    // sidebar rows) doesn't, so do it here for both paths.
                    if (e.button === 0) {
                      e.preventDefault()
                      e.stopPropagation()
                    }

                    // Dragging a SELECTED tab carries the whole selection as
                    // one block through the generic pane move — a multi-tab
                    // drag outranks the pane's own tab drag (the session drop
                    // language is single-session).
                    const dragSelection = selectionFor(node.id, shown, paneId)

                    if (dragSelection) {
                      startPaneDrag(
                        paneId,
                        e,
                        onTap,
                        stripRef.current ? { groupId: node.id, strip: stripRef.current } : undefined,
                        hideHeaderDoubleTap,
                        t.zones.tabCount(dragSelection.length),
                        dragSelection
                      )

                      return
                    }

                    // A pane may own its tab drag (a session tab speaks the
                    // session drop language — link/stack/split); `false` defers
                    // to the generic pane move (the workspace tab on a fresh
                    // draft has no session to link).
                    if (!chrome.tabDrag?.(e, onTap, hideHeaderDoubleTap)) {
                      startPaneDrag(
                        paneId,
                        e,
                        onTap,
                        stripRef.current ? { groupId: node.id, strip: stripRef.current } : undefined,
                        hideHeaderDoubleTap,
                        title
                      )
                    }
                  }}
                  role="tab"
                  selected={isSelected}
                  showCloseButton={chrome.showCloseButton !== false}
                  style={{ cursor: 'grab' }}
                >
                  {chrome.tabLead ? (
                    <span className="ml-2 -mr-1 flex shrink-0 items-center">{chrome.tabLead()}</span>
                  ) : null}
                  <PaneTabLabel>{tabLabel(paneId)}</PaneTabLabel>
                </PaneTab>
              )

              // A pane may wrap ITS tab in a domain menu (session verbs on a
              // tile tab); the wrapper needs the key since it's the root.
              return <Fragment key={paneId}>{chrome.tabWrap ? chrome.tabWrap(tab) : tab}</Fragment>
            })}

            {/* Plain "+" after the last tab of a CHAT strip (the workspace
                zone, or any zone holding session tabs) — always shown. Creates
                a new session tab (mirrors ⌘T) via the app-registered action;
                the pointerdown focuses this zone first, so the tab lands in
                THIS strip. Hidden when unwired or the zone is minimized. */}
            {shown.some(isSessionStripPane) && newSessionTabAction && !node.minimized && (
              <span
                // The action docks into the FOCUSED chat zone; clicking a
                // background strip's "+" must make THAT zone the focused one
                // first, or the tab opens in whichever zone was last clicked.
                // (pointerdown's own focus tracking would land after the click
                // handler reads the anchor.)
                onPointerDownCapture={() => noteActiveTreeGroup(node.id)}
              >
                <PaneStripGlyph
                  icon={<Codicon name="add" size="0.8125rem" />}
                  label={t.zones.newSessionTab}
                  onSelect={() => newSessionTabAction()}
                />
              </span>
            )}
          </PaneTabStrip>
        </ZoneMenu>
      )}

      {/* Body: the zone's pane content — the active pane and bounded hot-hidden
          cache stay mounted in absolute layers; parked panes are unmounted.
          `visibility` (not display) keeps the hidden pane's layout box, so
          scroll positions and measurements survive the round-trip — which also
          makes a hidden layer's rect identical to the visible one's, hence the
          marker document-wide lookups filter on (see pane-visibility.ts). */}
      {!node.minimized && (
        <div className="relative min-h-0 min-w-0 flex-1 overflow-hidden">
          {isEmpty ? (
            <div className="grid h-full place-items-center">
              {/* Same decode primitive as the CONNECTING boot overlay. */}
              <DecodeText className="text-(--ui-text-quaternary)" cursor prefix={1} text="HERMES" />
            </div>
          ) : (
            keptPanes.map(paneId => {
              const pane = paneFor(paneId)
              const isActive = paneId === activeId

              return (
                <div
                  aria-hidden={!isActive || undefined}
                  className={cn('absolute inset-0 overflow-auto', !isActive && 'pointer-events-none invisible')}
                  key={paneId}
                  {...hiddenPaneProps(!isActive)}
                >
                  {pane?.render ? (
                    // Visibility flows to the pane so a kept-alive chat surface
                    // can gate its hot (per-token) subscriptions while hidden;
                    // the group id identifies the ZONE it lives in, for state
                    // that is per-zone rather than per-tab (composer pop-out).
                    // The reload epoch keys the CONTENT, not this layer: a
                    // Reload remounts the contribution (effects re-run, state
                    // resets) while the layer — and every other tab — stays.
                    <PaneGroupContext.Provider value={node.id}>
                      <PaneLifecycleContext.Provider value={paneLifecycle[paneId]?.lifecycle ?? 'visible'}>
                        <PaneVisibleContext.Provider value={isActive}>
                          <ContribBoundary id={pane.id} key={paneEpochs[paneId] ?? 0}>
                            <ContribRender render={pane.render} />
                          </ContribBoundary>
                        </PaneVisibleContext.Provider>
                      </PaneLifecycleContext.Provider>
                    </PaneGroupContext.Provider>
                  ) : (
                    isActive && (
                      <div className="p-3 font-mono text-[11px] text-(--ui-text-quaternary)">
                        {t.zones.missingPane(paneId)}
                      </div>
                    )
                  )}
                </div>
              )
            })
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
            onPointerDown={e => startPaneDrag(activeId, e, undefined, undefined, undefined, active?.title ?? activeId)}
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
      <ZoneDropOverlay node={node} />
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

/**
 * The FancyZones drop overlay for one zone. Split out of TreeGroup so the
 * per-pointermove `$dropHint` churn re-renders only this lightweight node —
 * the zone's header, body, and menu-direction walk stay put during a drag.
 *
 * ONE dashed sheet per zone (DROP_SHEET_CLASS — the composer drop and the zone
 * targets speak identically): a quiet outline over every eligible zone,
 * accent-lit over the target, morphing to the hovered half for an edge split.
 */
function ZoneDropOverlay({ node }: { node: GroupNode }) {
  const dragging = useStore($treeDragging)
  const hint = useStore($dropHint)

  if (dragging === null) {
    return null
  }

  // A session drag (sidebar row) reuses this exact overlay — over ANY zone
  // that hosts a MAIN tile (stack into its tabs / split its edges); only a
  // CHAT zone's center is a link-to-chat (the composer overlay owns that
  // visual). Standing side chrome — the sidebar, files, terminal — hosts no
  // main tile, so a session can't land there: those zones stay DARK rather
  // than painting an idle outline the drop would only refuse. Same test
  // `tileZoneHost` (session-drag.ts) resolves the drop with, so what lights
  // up and what commits cannot disagree.
  const sessionDrag = dragging === SESSION_TILE_DRAG
  const chatZone = node.panes.some(isSessionStripPane)

  if (sessionDrag && !chatZone && !node.panes.some(isMainStripPane)) {
    return null
  }

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
  // the surface (ChatDropOverlay — the same sheet) owns that region; this sheet
  // fades out so the two never stack. A non-chat zone's center has no chat to
  // link, so it shows the normal stack sheet. Edges act like a tab.
  const centerLink = sessionDrag && primary && pos === 'center' && chatZone

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
          'absolute transition-[top,right,bottom,left,background-color,border-color,opacity] duration-150 ease-out',
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
      />
    </div>
  )
}
