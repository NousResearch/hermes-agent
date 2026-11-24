import { compactNumber } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { type ComponentProps, type CSSProperties, type MouseEvent, type ReactNode, useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { hudTargetSessionId } from '@/app/hud/handoff'
import { $widthClass } from '@/app/redesign/breakpoints'
import { $developerToolShown, toggleDeveloperToolPanel } from '@/app/redesign/developer-tool-panel'
import { PortalBandAgentSettings } from '@/app/redesign/portal-agent-settings'
import { togglePaneReveal } from '@/components/pane-shell'
import { toggleLayoutEditMode } from '@/components/pane-shell/edit-mode'
import { $narrowOverlayPaneIds } from '@/components/pane-shell/narrow-overlay-state'
import { $narrowViewport, resetLayoutTree } from '@/components/pane-shell/tree/store'
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
import { toggleHud } from '@/store/hud'
import { $interfaceMode, shownInMode, type Tiered } from '@/store/interface-mode'
import {
  $fileBrowserOpen,
  $leftSideOpen,
  $panesFlipped,
  CHAT_SIDEBAR_PANE_ID,
  FILE_BROWSER_PANE_ID,
  toggleLeftSide,
  togglePanesFlipped,
  toggleRightSide,
  toggleSidebarOpen
} from '@/store/layout'
import { REVIEW_PANE_ID } from '@/store/review'
import { $unreadSessionCount } from '@/store/session-dot-state'
import { usePortalShell } from '@/store/shell-variant'
import { $titlebarAppActionsSide, type BandToolId, TITLEBAR_FIXED_TOOLS } from '@/store/titlebar-app-actions'

import {
  $workspaceIsPage,
  type AppView,
  appViewForPath,
  hidesFixedTitlebarClusters,
  isOverlayView
} from '../routes'

import { $sidebarRevealSurface, useBandHiddenTools } from './band-tools'
import {
  $historyPosition,
  canGoBack,
  canGoForward,
  goBack,
  historyForward,
  useMouseHistoryButtons,
  useTrackHistoryPosition
} from './history-position'
import { PanelRevealButton } from './panel-reveal'
import { PortalSessionTab } from './portal-session-tab'
import { $openSheet, toggleShellSheet } from './shell-sheet'
import { ShellSheetScrim } from './shell-sheet-view'
import {
  TITLEBAR_CHROME_CHANGED_EVENT,
  TITLEBAR_ICON_BADGE_SCALE,
  titlebarButtonClass,
  titlebarIconSizeCss,
  titlebarToolClusterClass
} from './titlebar'
import { TitlebarIcon } from './titlebar-icon'
import { useTitlebarHeight } from './use-touch-titlebar'

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
  const portal = usePortalShell()
  const toolShown = useStore($developerToolShown)
  const rightOverlayId = overlayPanes.has(REVIEW_PANE_ID) ? REVIEW_PANE_ID : FILE_BROWSER_PANE_ID

  const edge = (id: string, dockedOpen: boolean, toggleDocked: () => void): SidebarEdge => {
    const open = overlayChrome ? overlayPanes.has(id) : dockedOpen
    const revealMode = open ? 'close' : 'open'

    return {
      open,
      expanded: browserHost ? open : undefined,
      toggle: overlayChrome
        ? () => togglePaneReveal(id, revealMode)
        : toggleDocked
    }
  }

  return {
    left: edge(CHAT_SIDEBAR_PANE_ID, leftSideOpen, toggleLeftSide),
    // The redesign's right side is its one tool panel, whichever zone stands outermost.
    right:
      portal && !narrow
        ? { open: toolShown, expanded: browserHost ? toolShown : undefined, toggle: toggleDeveloperToolPanel }
        : edge(rightOverlayId, fileBrowserOpen, toggleRightSide)
  }
}

/** Simple mode's floating way back to the hidden chat sidebar, when that is the reveal surface. */
function FloatingSidebarReveal() {
  const { t } = useI18n()

  return useStore($sidebarRevealSurface) === 'floating' ? (
    <PanelRevealButton label={t.simplifiedAgents.showChats} onReveal={toggleSidebarOpen} />
  ) : null
}

/** The band's sidebar button: the docked toggle, below wide the menu that opens the chat sidebar
 *  as a sheet, and on a compact page (no chat sidebar, no rail) the menu that opens the rail. */
function useBandMenuTool(
  portal: boolean,
  leftEdge: SidebarEdge
): Pick<TitlebarTool, 'expanded' | 'hidden' | 'icon' | 'onSelect'> {
  const hidden = useBandHiddenTools().has('sidebar')
  const compact = useStore($widthClass) === 'compact' && portal
  const workspacePage = useStore($workspaceIsPage)
  const openSheet = useStore($openSheet)
  const navSheet = compact && workspacePage

  return {
    expanded: navSheet ? openSheet === 'nav' : leftEdge.expanded,
    hidden,
    // Folded, the redesign's button shows the glyph of what it does: expand, not collapse.
    icon: (
      <TitlebarIcon
        name={compact ? 'menu' : portal && !leftEdge.open ? 'layout-sidebar-left-off' : 'layout-sidebar-left'}
      />
    ),
    onSelect: () => {
      triggerHaptic('tap')

      if (navSheet) {
        toggleShellSheet('nav')
      } else {
        leftEdge.toggle()
      }
    }
  }
}

/** The redesigned band leads with Back and Forward on every screen (Figma 906:19467, 856:2713):
 *  the window's own router history. A workspace page with nothing behind it still has a way back
 *  (`goBack`'s fallback). `$bandHiddenTools` drops them where the band has no room or a browser
 *  toolbar already carries them. */
function usePortalHistoryTools(bandHidden: ReadonlySet<BandToolId>): TitlebarTool[] {
  const { t } = useI18n()
  const navigate = useNavigate()
  const position = useStore($historyPosition)
  const workspacePage = useStore($workspaceIsPage)

  return [
    {
      disabled: !canGoBack(position) && !workspacePage,
      hidden: bandHidden.has('history-back'),
      icon: <TitlebarIcon name="arrow-left" />,
      id: 'history-back',
      label: t.common.back,
      onSelect: () => goBack(navigate)
    },
    {
      disabled: !canGoForward(position),
      hidden: bandHidden.has('history-forward'),
      icon: <TitlebarIcon name="arrow-right" />,
      id: 'history-forward',
      label: t.preview.web.goForward,
      onSelect: historyForward
    }
  ]
}

/** The redesign centres its 30px chrome buttons in its 60px band; native controls are centred
 *  there too, so the pre-Tahoe optical nudge stands down. The leading cluster starts at
 *  `--portal-band-start` (portalBandLeadingInset: past macOS traffic lights, else on the content inset).
 *  Classic keeps the stylesheet's placement. */
function usePortalClusterStyles(portal: boolean): { clusterStyle?: CSSProperties; leftClusterStyle?: CSSProperties } {
  const titlebarHeight = useTitlebarHeight()

  if (!portal) {
    return {}
  }

  const clusterStyle = {
    '--titlebar-controls-top': `calc((${titlebarHeight}px - var(--titlebar-control-height)) / 2)`,
    '--titlebar-controls-y-nudge': '0px'
  } as CSSProperties

  return { clusterStyle, leftClusterStyle: { ...clusterStyle, left: 'var(--portal-band-start)' } }
}

export function TitlebarControls({ leftTools = [], tools = [], onOpenSettings }: TitlebarControlsProps) {
  const portal = usePortalShell()
  const { t } = useI18n()
  const navigate = useNavigate()
  const location = useLocation()
  const modHeld = useModifierHeld()
  const panesFlipped = useStore($panesFlipped)
  const unreadCount = useStore($unreadSessionCount)
  const appActionsSide = useStore($titlebarAppActionsSide)
  const interfaceMode = useStore($interfaceMode)
  const unreadBadge = unreadCount > 0 ? unreadCount : undefined
  const unreadHint = unreadBadge ? ` · ${t.titlebar.unreadSessions(unreadBadge)}` : ''
  // One filter for every cluster: a tool's own `hidden`, then the mode's tier.
  const shown = shownInMode(interfaceMode)
  const visibleTool = (tool: TitlebarTool) => !tool.hidden && shown(tool)

  const { pageOwnsTitlebar, posture } = useBandPosture(portal)

  // POSITIONAL toggles: each button shows/hides everything on its physical
  // side of the main zone (the layout tree collapses the whole side), so they
  // stay correct through flips and rearranges. Both edges resolve their column
  // from the live tree (see toggleLeftSide / toggleRightSide) — the browser
  // column, the sessions column, whatever is physically left / right. Never an
  // active highlight — plain show/hide affordances.
  const { left: leftEdge, right: rightEdge } = useSidebarEdges()
  const leftLabel = leftEdge.open ? t.titlebar.hideSidebar : t.titlebar.showSidebar
  const rightLabel = rightEdge.open ? t.titlebar.hideRightSidebar : t.titlebar.showRightSidebar
  const bandHidden = useBandHiddenTools()
  const historyTools = usePortalHistoryTools(bandHidden)
  useTrackHistoryPosition()
  useMouseHistoryButtons(portal, navigate)

  const sidebarTool: TitlebarTool = {
    ...TITLEBAR_FIXED_TOOLS.sidebar,
    ...useBandMenuTool(portal, leftEdge),
    actionId: 'view.toggleSidebar',
    badge: panesFlipped ? undefined : unreadBadge,
    id: 'sidebar',
    label: `${leftLabel}${panesFlipped ? '' : unreadHint}`
  }

  const flipTool: TitlebarTool = {
    ...TITLEBAR_FIXED_TOOLS['flip-panes'],
    actionId: 'view.flipPanes',
    // A compact window has one column: nothing to swap.
    hidden: bandHidden.has('flip-panes'),
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
      hidden: bandHidden.has('settings'),
      label: t.titlebar.openSettings,
      onSelect: () => {
        triggerHaptic('open')
        onOpenSettings()
      }
    },
    {
      ...TITLEBAR_FIXED_TOOLS.layout,
      hidden: bandHidden.has('layout'),
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
      hidden: bandHidden.has('hud'),
      icon: <TitlebarIcon name="comment-discussion" />,
      id: 'hud',
      label: t.titlebar.enterHud,
      onSelect: () => {
        triggerHaptic('open')
        toggleHud(hudTargetSessionId())
      }
    }
  ]

  const { clusterStyle, leftClusterStyle } = usePortalClusterStyles(portal)

  // Route changes can replace measured clusters without resizing the panels.
  useEffect(() => {
    window.dispatchEvent(new CustomEvent(TITLEBAR_CHROME_CHANGED_EVENT))
  }, [location.pathname, pageOwnsTitlebar])

  // Overlays own the window. These clusters are `fixed` at a higher z-index
  // than the overlay card, so they'd otherwise bleed over it — hide them (and
  // the nested titleBar slots) and let the overlay's own chrome take over.
  // Classic unmounts them there, as it always has. In the redesign Agents,
  // Schedules and Settings are primary destinations, so the clusters are
  // parked instead (hidden, same tree): `titleBar.*` slots are permanent mount
  // points, and a plugin's setup/cleanup must not run on every visit.
  const parked = posture === 'parked'
  const parkedProps = parked ? PARKED_CLUSTER : {}

  if (posture === 'gone') {
    return null
  }

  const leftClusterClass = cn(
    titlebarToolClusterClass,
    'left-(--titlebar-controls-left) top-(--titlebar-controls-top) translate-y-(--titlebar-controls-y-nudge)'
  )

  // A contributed full page (`extension`) yields the fixed clusters only while
  // it actually projects chrome into the band — page-mounted `titleBar.*` slots
  // like kanban's board switcher. A page that mounts no titlebar chrome keeps
  // the app's controls; an empty claim would leave a bare strip on every plugin
  // route. Contributed `titleBar.tools` items keep rendering here too, so a
  // chrome-owning page never silently drops a registered item.
  if (posture === 'page') {
    const pageTools = [...leftTools, ...tools].filter(visibleTool)

    // Both markers are required even when a page contributes to only one side.
    return (
      <>
        <div className={leftClusterClass} data-titlebar-cluster="left" style={leftClusterStyle}>
          {pageTools.map(tool => (
            <TitlebarToolButton key={tool.id} navigate={navigate} portal={portal} tool={tool} />
          ))}
          <Slot area="titleBar.left" />
        </div>
        <div
          className={cn(titlebarToolClusterClass, 'right-(--titlebar-tools-right) top-(--titlebar-controls-top)')}
          data-titlebar-cluster="right"
          style={clusterStyle}
        >
          <Slot area="titleBar.right" />
        </div>
      </>
    )
  }

  const visibleLeftTools = (
    appActionsSide === 'left'
      ? [...historyTools, sidebarTool, ...systemTools, ...leftTools]
      : [...historyTools, sidebarTool, ...leftTools]
  ).filter(visibleTool)

  const visibleSystemTools = appActionsSide === 'right' ? systemTools.filter(visibleTool) : []
  const visiblePaneTools = tools.filter(visibleTool)
  const visibleRightFixedTools = [flipTool, rightSidebarTool].filter(visibleTool)

  return (
    <>
      <div
        aria-label={t.shell.windowControls}
        className={leftClusterClass}
        data-titlebar-cluster="left"
        {...parkedProps}
        style={leftClusterStyle}
      >
        {visibleLeftTools.map(tool => (
          <TitlebarToolButton key={tool.id} navigate={navigate} portal={portal} tool={tool} />
        ))}
        <Slot area="titleBar.left" />
        <Slot area="titleBar.center" />
      </div>

      {!parked && (
        <BandOverlays
          clusterStyle={clusterStyle}
          navigate={navigate}
          paneTools={visiblePaneTools}
          portal={portal}
        />
      )}

      <div
        aria-label={t.shell.appControls}
        className={cn(titlebarToolClusterClass, 'right-(--titlebar-tools-right) top-(--titlebar-controls-top)')}
        data-titlebar-cluster="right"
        {...parkedProps}
        style={clusterStyle}
      >
        {visibleSystemTools.map(tool => (
          <TitlebarToolButton key={tool.id} navigate={navigate} portal={portal} tool={tool} />
        ))}
        {visibleRightFixedTools.map(tool => (
          <TitlebarToolButton key={tool.id} navigate={navigate} portal={portal} tool={tool} />
        ))}
        <PortalBandAgentSettings />
        <Slot area="titleBar.right" />
      </div>
    </>
  )
}

/** What the band shows for a view: nothing (a classic overlay), its clusters parked (a redesign
 *  overlay destination), only a contributed page's own chrome, or the full band. */
function bandPosture(view: AppView, portal: boolean, pageOwnsTitlebar: boolean): 'full' | 'gone' | 'page' | 'parked' {
  if (isOverlayView(view)) {
    return portal ? 'parked' : 'gone'
  }

  return hidesFixedTitlebarClusters(view) && pageOwnsTitlebar ? 'page' : 'full'
}

function useBandPosture(portal: boolean) {
  const location = useLocation()
  // `titleBar.*` slot content is mount-scoped — a page's <Contribute> registers
  // only while that surface is up — so a non-empty area means a page is
  // actively projecting chrome into the band right now.
  const titleBarLeft = useContributions('titleBar.left')
  const titleBarRight = useContributions('titleBar.right')
  const pageOwnsTitlebar = titleBarLeft.length + titleBarRight.length > 0

  return { pageOwnsTitlebar, posture: bandPosture(appViewForPath(location.pathname), portal, pageOwnsTitlebar) }
}

/** The redesign's session tab. It sits mid-band, over the chat pane's window-drag handle, and
 *  Electron folds app-region rects in document order with no regard for stacking: a drag rect
 *  that comes later re-covers an earlier no-drag one, and macOS then drags the window instead of
 *  delivering the click. So this mounts AFTER the pane tree, where its "+" is carved out last. */
export function TitlebarSessionTab() {
  const portal = usePortalShell()
  const { posture } = useBandPosture(portal)
  const { clusterStyle } = usePortalClusterStyles(portal)

  if (!portal || posture !== 'full') {
    return null
  }

  return (
    <div
      className={cn(titlebarToolClusterClass, 'portal-session-tab-cluster top-(--titlebar-controls-top)')}
      style={clusterStyle}
    >
      <PortalSessionTab />
    </div>
  )
}

/** A parked cluster: out of sight and out of the tab order, still mounted. */
const PARKED_CLUSTER = { hidden: true, inert: true } as const

interface BandOverlaysProps {
  clusterStyle: CSSProperties | undefined
  navigate: ReturnType<typeof useNavigate>
  paneTools: TitlebarTool[]
  portal: boolean
}

/** The band's chrome besides its two clusters: the redesign's sheet scrim, the floating sidebar
 *  reveal, and the pane tools. None of it holds a plugin mount, so a parked band
 *  drops it outright. */
function BandOverlays({ clusterStyle, navigate, paneTools, portal }: BandOverlaysProps) {
  const { t } = useI18n()

  return (
    <>
      {portal && <ShellSheetScrim />}
      <FloatingSidebarReveal />
      {paneTools.length > 0 && (
        <div
          aria-label={t.shell.appControls}
          className={cn(
            titlebarToolClusterClass,
            'top-[calc(var(--titlebar-controls-top)+var(--right-rail-top-inset,0px))] right-[calc(var(--titlebar-tools-right)+var(--shell-preview-toolbar-gap,0))]'
          )}
          style={clusterStyle}
        >
          {paneTools.map(tool => (
            <TitlebarToolButton key={tool.id} navigate={navigate} portal={portal} tool={tool} />
          ))}
        </div>
      )}
    </>
  )
}

interface TitlebarToolButtonProps {
  navigate: ReturnType<typeof useNavigate>
  /** The redesign draws every titlebar tool as a 30px chrome button. */
  portal: boolean
  tool: TitlebarTool
}

function TitlebarToolButton({ navigate, portal, tool }: TitlebarToolButtonProps) {
  // Classic titlebar actions never show an active background — state reads
  // from the icon itself (e.g. the mute/unmute glyph). aria-pressed still
  // carries it for a11y.
  const className = cn(
    portal ? 'portal-chrome-button' : [titlebarButtonClass, 'bg-transparent'],
    'select-none',
    tool.className
  )

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
