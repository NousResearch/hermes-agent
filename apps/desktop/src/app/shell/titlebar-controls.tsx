import { compactNumber } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { type ComponentProps, type MouseEvent, type ReactNode, useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { hudTargetSessionId } from '@/app/hud/handoff'
import { PANE_TOGGLE_REVEAL_EVENT } from '@/components/pane-shell'
import { toggleLayoutEditMode } from '@/components/pane-shell/edit-mode'
import { $narrowOverlayPaneIds } from '@/components/pane-shell/narrow-overlay-state'
import { $mainZoneLeft } from '@/components/pane-shell/tree/main-zone-anchor'
import { $narrowViewport, resetLayoutTree, revealTreePane } from '@/components/pane-shell/tree/store'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tip, TipKeybindLabel } from '@/components/ui/tooltip'
import { Slot } from '@/contrib/react/slot'
import { useContributions } from '@/contrib/react/use-contributions'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { hostHas } from '@/lib/host-capabilities'
import { formatModifierToken } from '@/lib/keybinds/combo'
import { cn } from '@/lib/utils'
import { recordAction } from '@/store/desktop-metrics'
import { canUseHud, toggleHud } from '@/store/hud'
import { $interfaceMode, shownInMode, type Tiered } from '@/store/interface-mode'
import {
  $fileBrowserOpen,
  $leftSideOpen,
  $panesFlipped,
  CHAT_SIDEBAR_PANE_ID,
  FILE_BROWSER_PANE_ID,
  toggleLeftSide,
  togglePanesFlipped,
  toggleRightSide
} from '@/store/layout'
import { REVIEW_PANE_ID } from '@/store/review'
import { $unreadSessionCount } from '@/store/session-dot-state'
import { $titlebarAppActionsSide, TITLEBAR_FIXED_TOOLS } from '@/store/titlebar-app-actions'
import { $canGoBack, backVisible, goBackToPreviousView } from '@/store/view-history'

import { appViewForPath, hidesFixedTitlebarClusters, isOverlayView } from '../routes'

import {
  TITLEBAR_CHROME_CHANGED_EVENT,
  TITLEBAR_ICON_BADGE_SCALE,
  titlebarButtonClass,
  titlebarIconSizeCss,
  titlebarToolClusterClass
} from './titlebar'
import { TitlebarIcon } from './titlebar-icon'

export interface TitlebarTool extends Tiered {
  id: string
  label: string
  active?: boolean
  expanded?: boolean
  className?: string
  disabled?: boolean
  hidden?: boolean
  href?: string
  icon: ReactNode
  onSelect?: (event?: MouseEvent) => void
  /** Keybind action id — when set, the tooltip shows the label + keybind hint. */
  actionId?: string
  /** Overlay count on the glyph (unread sessions). Hidden when 0/undefined. */
  badge?: number
  title?: string
  to?: string
  /** Durable `data-tour` handle. Tools are addressed by icon and translated
   *  label otherwise, and neither survives a theme or a locale change. */
  tour?: string
}

export type TitlebarToolSide = 'left' | 'right'
export type SetTitlebarToolGroup = (id: string, tools: readonly TitlebarTool[], side?: TitlebarToolSide) => void

interface TitlebarControlsProps extends ComponentProps<'div'> {
  leftTools?: readonly TitlebarTool[]
  tools?: readonly TitlebarTool[]
  onOpenSettings: () => void
}

/**
 * The layout button's glyph. Morphs into its composite reset form — the
 * layout icon wearing a small counter-clockwise arrow badge ("layout, back
 * to how it was") — ONLY while the pointer is on the button AND ⌘/Ctrl is
 * held: hover gates via CSS (`group/tool` on the button), the modifier via
 * the window listener. Pressing the modifier elsewhere changes nothing.
 */
function LayoutGlyph({ modHeld }: { modHeld: boolean }) {
  return (
    <>
      <span className={cn('inline-flex', modHeld && 'group-hover/tool:hidden')}>
        <TitlebarIcon name="layout" />
      </span>
      <span className={cn('relative hidden', modHeld && 'group-hover/tool:inline-flex')}>
        <TitlebarIcon name="layout" />
        <span className="absolute -bottom-1 -right-1.5 grid place-items-center rounded-full bg-(--ui-bg-chrome) p-px">
          <TitlebarIcon className="-scale-x-100" name="refresh" size={titlebarIconSizeCss(TITLEBAR_ICON_BADGE_SCALE)} />
        </span>
      </span>
    </>
  )
}

/** Overlay count on a titlebar glyph. Hidden when count is 0/undefined. */
function withCountBadge(icon: ReactNode, count: number | undefined): ReactNode {
  if (!count) {
    return icon
  }

  return (
    <span className="relative inline-flex">
      {icon}
      <span className="pointer-events-none absolute -top-2.5 -right-1.5 z-1">
        <Badge aria-hidden size="overlay" variant="solid">
          {compactNumber(count)}
        </Badge>
      </span>
    </span>
  )
}

/** Live ⌘/Ctrl tracking — mod-click affordances telegraph themselves (the
 *  layout button morphs into its reset form while the modifier is down). */
function useModifierHeld(): boolean {
  const [held, setHeld] = useState(false)

  useEffect(() => {
    const sync = (event: KeyboardEvent) => setHeld(event.metaKey || event.ctrlKey)
    const clear = () => setHeld(false)

    window.addEventListener('keydown', sync)
    window.addEventListener('keyup', sync)
    window.addEventListener('blur', clear)

    return () => {
      window.removeEventListener('keydown', sync)
      window.removeEventListener('keyup', sync)
      window.removeEventListener('blur', clear)
    }
  }, [])

  return held
}

interface SidebarEdge {
  /** Whether the side reads as open, which picks the button's Show or Hide label. */
  open: boolean
  /** The button's aria-expanded; only the browser-hosted Desktop reports it. */
  expanded?: boolean
  toggle: () => void
}

/** The titlebar's two sidebar buttons. The browser-hosted Desktop's narrow layout has no hover to
 * reveal by, so there the buttons read and drive the overlay itself, and they send an explicit open
 * or close: a visible hover reveal is not pinned, and a generic toggle would pin it rather than
 * perform the labelled Hide. A narrow native window keeps the docked toggles, which already route
 * through the overlay's toggle. */
function useSidebarEdges(): { left: SidebarEdge; right: SidebarEdge } {
  const fileBrowserOpen = useStore($fileBrowserOpen)
  const leftSideOpen = useStore($leftSideOpen)
  const narrow = useStore($narrowViewport)
  const overlayPanes = useStore($narrowOverlayPaneIds)
  const browserHost = hostHas('narrowOverlayChrome')
  const overlayChrome = narrow && browserHost
  const rightOverlayId = overlayPanes.has(REVIEW_PANE_ID) ? REVIEW_PANE_ID : FILE_BROWSER_PANE_ID

  const edge = (id: string, dockedOpen: boolean, toggleDocked: () => void): SidebarEdge => {
    const open = overlayChrome ? overlayPanes.has(id) : dockedOpen
    const revealMode = open ? 'close' : 'open'

    return {
      open,
      expanded: browserHost ? open : undefined,
      toggle: overlayChrome
        ? () => window.dispatchEvent(new CustomEvent(PANE_TOGGLE_REVEAL_EVENT, { detail: { id, mode: revealMode } }))
        : toggleDocked
    }
  }

  return {
    left: edge(CHAT_SIDEBAR_PANE_ID, leftSideOpen, toggleLeftSide),
    right: edge(rightOverlayId, fileBrowserOpen, toggleRightSide)
  }
}

export function TitlebarControls({ leftTools = [], tools = [], onOpenSettings }: TitlebarControlsProps) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const location = useLocation()
  const modHeld = useModifierHeld()
  const panesFlipped = useStore($panesFlipped)
  const unreadCount = useStore($unreadSessionCount)
  const appActionsSide = useStore($titlebarAppActionsSide)
  const interfaceMode = useStore($interfaceMode)
  const canGoBack = useStore($canGoBack)
  const mainZoneLeft = useStore($mainZoneLeft)
  const unreadBadge = unreadCount > 0 ? unreadCount : undefined
  const unreadHint = unreadBadge ? ` · ${t.titlebar.unreadSessions(unreadBadge)}` : ''
  // One filter for every cluster: a tool's own `hidden`, then the mode's tier.
  const shown = shownInMode(interfaceMode)
  const visibleTool = (tool: TitlebarTool) => !tool.hidden && shown(tool)

  // `titleBar.*` slot content is mount-scoped — a page's <Contribute> registers
  // only while that surface is up — so a non-empty area means a page is
  // actively projecting chrome into the band right now.
  const titleBarLeft = useContributions('titleBar.left')
  const titleBarRight = useContributions('titleBar.right')
  const pageOwnsTitlebar = titleBarLeft.length + titleBarRight.length > 0

  // POSITIONAL toggles: each button shows/hides everything on its physical
  // side of the main zone (the layout tree collapses the whole side), so they
  // stay correct through flips and rearranges. Both edges resolve their column
  // from the live tree (see toggleLeftSide / toggleRightSide) — the browser
  // column, the sessions column, whatever is physically left / right. Never an
  // active highlight — plain show/hide affordances.
  const { left: leftEdge, right: rightEdge } = useSidebarEdges()
  const leftLabel = leftEdge.open ? t.titlebar.hideSidebar : t.titlebar.showSidebar
  const rightLabel = rightEdge.open ? t.titlebar.hideRightSidebar : t.titlebar.showRightSidebar

  const sidebarTool: TitlebarTool = {
    ...TITLEBAR_FIXED_TOOLS.sidebar,
    actionId: 'view.toggleSidebar',
    badge: panesFlipped ? undefined : unreadBadge,
    icon: <TitlebarIcon name="layout-sidebar-left" />,
    id: 'sidebar',
    expanded: leftEdge.expanded,
    label: `${leftLabel}${panesFlipped ? '' : unreadHint}`,
    onSelect: () => {
      triggerHaptic('tap')
      leftEdge.toggle()
    }
  }

  const flipTool: TitlebarTool = {
    ...TITLEBAR_FIXED_TOOLS['flip-panes'],
    actionId: 'view.flipPanes',
    icon: <TitlebarIcon name="arrow-swap" />,
    id: 'flip-panes',
    label: t.titlebar.swapSidebarSides,
    onSelect: () => {
      triggerHaptic('tap')
      togglePanesFlipped()
    }
  }

  const rightSidebarTool: TitlebarTool = {
    ...TITLEBAR_FIXED_TOOLS['right-sidebar'],
    actionId: 'view.toggleRightSidebar',
    badge: panesFlipped ? unreadBadge : undefined,
    icon: <TitlebarIcon name="layout-sidebar-right" />,
    id: 'right-sidebar',
    expanded: rightEdge.expanded,
    label: `${rightLabel}${panesFlipped ? unreadHint : ''}`,
    onSelect: () => {
      triggerHaptic('tap')
      rightEdge.toggle()
    },
    tour: 'right-pane-toggle'
  }

  // Static system tools — always pinned to the screen's right edge so the
  // left titlebar stays free for tabs (#107351).
  const systemTools: TitlebarTool[] = [
    {
      ...TITLEBAR_FIXED_TOOLS.settings,
      actionId: 'nav.settings',
      icon: <TitlebarIcon name="settings-gear" />,
      id: 'settings',
      label: t.titlebar.openSettings,
      onSelect: () => {
        triggerHaptic('open')
        onOpenSettings()
      }
    },
    {
      ...TITLEBAR_FIXED_TOOLS.layout,
      className: 'group/tool',
      // Hover + held ⌘/Ctrl morphs the glyph into its reset form (see
      // LayoutGlyph) — the mod-click telegraphs itself before it happens.
      icon: <LayoutGlyph modHeld={modHeld} />,
      id: 'layout',
      label: t.titlebar.layoutEditor,
      onSelect: event => {
        if (event?.metaKey || event?.ctrlKey) {
          triggerHaptic('warning')
          resetLayoutTree()

          return
        }

        triggerHaptic('open')
        toggleLayoutEditMode()
      },
      title: t.titlebar.layoutEditorTitle(formatModifierToken('mod'))
    },
    {
      ...TITLEBAR_FIXED_TOOLS.hud,
      // No `title`: TitlebarToolButton passes `title` to TipKeybindLabel as a
      // text OVERRIDE, so a long sentence there replaces the short label and
      // crowds the ⌘⇧H hint off the tooltip. Label only — the hint is appended
      // from the action registry, same as every other tool here.
      actionId: 'view.toggleHud',
      hidden: !canUseHud(),
      icon: <TitlebarIcon name="comment-discussion" />,
      id: 'hud',
      label: t.titlebar.enterHud,
      onSelect: () => {
        triggerHaptic('open')
        toggleHud(hudTargetSessionId())
      }
    }
  ]

  const view = appViewForPath(location.pathname)
  // One rule for the affordance, read here and inside BackViewButton: the
  // overlay branch needs to know whether its lone cluster exists at all.
  const backShown = backVisible(view, canGoBack)

  // Route changes can replace measured clusters without resizing the panels,
  // and the anchored back control is a measurement input for the zone's own
  // band (panel-titlebar.ts reserves its width): it mounts late — once
  // main-zone-anchor.ts publishes where the zone starts — and moves with the
  // zone, so the band has to re-read it.
  useEffect(() => {
    window.dispatchEvent(new CustomEvent(TITLEBAR_CHROME_CHANGED_EVENT))
  }, [location.pathname, pageOwnsTitlebar, backShown, mainZoneLeft])

  const leftClusterClass = cn(
    titlebarToolClusterClass,
    'left-(--titlebar-controls-left) top-(--titlebar-controls-top) translate-y-(--titlebar-controls-y-nudge)'
  )

  // Overlays own the window. These clusters are `fixed` at a higher z-index
  // than the overlay card, so they'd otherwise bleed over it — hide them (and
  // the nested titleBar slots) and let the overlay's own chrome take over.
  // Back is the one exception: the card starts BELOW the titlebar band, so a
  // lone button there covers nothing, and it is the way out of an overlay. It
  // carries no `data-titlebar-cluster` marker, so the band-measurement hooks
  // read exactly what they read today.
  // An overlay CARD starts below the band and covers the zone, so the way out
  // keeps the window's own left edge there; on a page route it anchors to the
  // main view instead (see the page cluster below).
  if (isOverlayView(view)) {
    return backShown ? (
      <div className={leftClusterClass}>
        <BackViewButton />
      </div>
    ) : null
  }

  // The whole reason this stands apart from the left cluster: it belongs to the
  // MAIN VIEW — the zone a page renders in — not to the window. So it is
  // anchored where that zone starts, and clamped clear of the window controls
  // when the sidebar is too narrow to leave room (main-zone-anchor.ts). No
  // `data-titlebar-cluster` marker: the band-measurement hooks must not read it
  // as chrome competing with the page's own band.
  const backCluster =
    backShown && mainZoneLeft !== null ? (
      <div
        className={cn(
          titlebarToolClusterClass,
          'top-(--titlebar-controls-top) translate-y-(--titlebar-controls-y-nudge)'
        )}
        data-titlebar-back=""
        style={{ left: `${mainZoneLeft}px` }}
      >
        <BackViewButton />
      </div>
    ) : null

  // A contributed full page (`extension`) yields the fixed clusters only while
  // it actually projects chrome into the band — page-mounted `titleBar.*` slots
  // like kanban's board switcher. A page that mounts no titlebar chrome keeps
  // the app's controls; an empty claim would leave a bare strip on every plugin
  // route. Contributed `titleBar.tools` items keep rendering here too, so a
  // chrome-owning page never silently drops a registered item.
  if (hidesFixedTitlebarClusters(view) && pageOwnsTitlebar) {
    const pageTools = [...leftTools, ...tools].filter(visibleTool)

    // Both markers are required even when a page contributes to only one side.
    return (
      <>
        {backCluster}
        <div className={leftClusterClass} data-titlebar-cluster="left">
          {pageTools.map(tool => (
            <TitlebarToolButton key={tool.id} navigate={navigate} tool={tool} />
          ))}
          <Slot area="titleBar.left" />
        </div>
        <div
          className={cn(titlebarToolClusterClass, 'right-(--titlebar-tools-right) top-(--titlebar-controls-top)')}
          data-titlebar-cluster="right"
        >
          <Slot area="titleBar.right" />
        </div>
      </>
    )
  }

  const visibleLeftTools = [
    ...(appActionsSide === 'left' ? [sidebarTool, ...systemTools, ...leftTools] : [sidebarTool, ...leftTools])
  ].filter(visibleTool)

  const visibleSystemTools = appActionsSide === 'right' ? systemTools.filter(visibleTool) : []
  const visiblePaneTools = tools.filter(visibleTool)
  const visibleRightFixedTools = [flipTool, rightSidebarTool].filter(visibleTool)

  return (
    <>
      {backCluster}
      <div aria-label={t.shell.windowControls} className={leftClusterClass} data-titlebar-cluster="left">
        {visibleLeftTools.map(tool => (
          <TitlebarToolButton key={tool.id} navigate={navigate} tool={tool} />
        ))}
        <Slot area="titleBar.left" />
        <Slot area="titleBar.center" />
      </div>

      {visiblePaneTools.length > 0 && (
        <div
          aria-label={t.shell.appControls}
          className={cn(
            titlebarToolClusterClass,
            'top-[calc(var(--titlebar-controls-top)+var(--right-rail-top-inset,0px))] right-[calc(var(--titlebar-tools-right)+var(--shell-preview-toolbar-gap,0))]'
          )}
        >
          {visiblePaneTools.map(tool => (
            <TitlebarToolButton key={tool.id} navigate={navigate} tool={tool} />
          ))}
        </div>
      )}

      <div
        aria-label={t.shell.appControls}
        className={cn(titlebarToolClusterClass, 'right-(--titlebar-tools-right) top-(--titlebar-controls-top)')}
        data-titlebar-cluster="right"
      >
        {visibleSystemTools.map(tool => (
          <TitlebarToolButton key={tool.id} navigate={navigate} tool={tool} />
        ))}
        {visibleRightFixedTools.map(tool => (
          <TitlebarToolButton key={tool.id} navigate={navigate} tool={tool} />
        ))}
        <Slot area="titleBar.right" />
      </div>
    </>
  )
}

/**
 * The way back out of a view — the same glyph, label and handler wherever it
 * stands: alone in the band over an overlay card, and in the page cluster
 * anchored to the main view on a page route. Both hosts render it as-is, so the
 * affordance cannot drift between them.
 */
export function BackViewButton() {
  const { t } = useI18n()
  const navigate = useNavigate()
  const location = useLocation()
  const canGoBack = useStore($canGoBack)
  const view = appViewForPath(location.pathname)

  if (!backVisible(view, canGoBack)) {
    return null
  }

  return (
    <TitlebarToolButton
      navigate={navigate}
      tool={{
        icon: <TitlebarIcon name="arrow-left" />,
        id: 'back',
        label: t.titlebar.back,
        onSelect: () => {
          if (goBackToPreviousView(navigate, revealTreePane)) {
            triggerHaptic('tap')
          }
        }
      }}
    />
  )
}

function TitlebarToolButton({ navigate, tool }: { navigate: ReturnType<typeof useNavigate>; tool: TitlebarTool }) {
  // Titlebar actions never show an active background — state reads from the
  // icon itself (e.g. the mute/unmute glyph). aria-pressed still carries it
  // for a11y.
  const className = cn(titlebarButtonClass, 'bg-transparent select-none', tool.className)

  const tooltipLabel = tool.actionId ? (
    <TipKeybindLabel actionId={tool.actionId} text={tool.title ?? tool.label} />
  ) : (
    (tool.title ?? tool.label)
  )

  if (tool.href) {
    return (
      <Tip label={tooltipLabel} placement="toolbar">
        <Button asChild className={className} size="icon-titlebar" variant="ghost">
          <a
            aria-label={tool.label}
            data-tour={tool.tour}
            href={tool.href}
            onPointerDown={event => event.stopPropagation()}
            rel="noreferrer"
            target="_blank"
          >
            {withCountBadge(tool.icon, tool.badge)}
          </a>
        </Button>
      </Tip>
    )
  }

  return (
    <Tip label={tooltipLabel} placement="toolbar">
      <Button
        aria-expanded={tool.expanded}
        aria-label={tool.label}
        aria-pressed={tool.active ?? undefined}
        className={className}
        data-tour={tool.tour}
        disabled={tool.disabled}
        onClick={event => {
          if (tool.actionId) {
            recordAction(tool.actionId, 'click')
          }

          if (tool.to) {
            navigate(tool.to)
          }

          tool.onSelect?.(event)
        }}
        onPointerDown={event => event.stopPropagation()}
        size="icon-titlebar"
        type="button"
        variant="ghost"
      >
        {withCountBadge(tool.icon, tool.badge)}
      </Button>
    </Tip>
  )
}
