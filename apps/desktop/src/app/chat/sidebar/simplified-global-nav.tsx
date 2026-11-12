import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'
import { useLocation } from 'react-router'

import wing from '@/assets/catalog-wing.svg'
import figmaRobot from '@/assets/figma-agent-robot.svg'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from '@/components/ui/context-menu'
import { Tip, TipKeybindLabel } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { recordIaAction } from '@/store/ia-metrics'
import type { IaAction } from '@/store/ia-metrics-contract'
import { $sidebarOpen, setSidebarOpen, toggleSidebarOpen } from '@/store/layout'
import { $focusedSessionIsTile, $focusedStoredSessionId } from '@/store/session-focus'

import { $pageHasSidebar, togglePageSidebar } from '../../overlays/page-sidebar'
import { preloadRailPage } from '../../page-views'
import { $widthClass } from '../../redesign/breakpoints'
import {
  appViewForPath,
  ARTIFACTS_ROUTE,
  CAPABILITIES_ROUTE,
  CRON_ROUTE,
  NEW_CHAT_ROUTE,
  PROFILES_ROUTE,
  sessionRoute,
  SETTINGS_ROUTE
} from '../../routes'
import { REDESIGN_SETTINGS_SECTIONS } from '../../settings/redesign-navigation'
import { $openSheet } from '../../shell/shell-sheet'
import type { SidebarNavItem } from '../../types'

import { useIaNavigationMetrics } from './use-ia-navigation-metrics'
import { useSimplifiedPluginPages } from './use-simplified-plugin-pages'

interface SimplifiedGlobalNavProps {
  onNavigate: (item: SidebarNavItem) => void
  footer?: ReactNode
}

/** Destinations that open with a panel beside the stage: the chat sidebar, or a page's sidebar panel
 *  (Agents and an agent's pages, Schedules, Settings). They share one open state (page-sidebar.tsx). */
const PANEL_AREAS = new Set(['chat', 'profiles', 'cron', 'settings'])

interface PanelToggle {
  shown: boolean
  toggle: () => void
  label: string
}

/** The open destination's own panel, which its rail button folds and brings back (Figma 1054:20985,
 *  1054:21668), so the way to collapse never moves: Chats toggles the chat sidebar, a page with a
 *  panel toggles that panel. Docked on a wide window, a sheet below it. Null when the destination is
 *  not open or has no panel. */
function usePanelToggle(): (area: string, active: boolean) => null | PanelToggle {
  const { t } = useI18n()
  const sidebarOpen = useStore($sidebarOpen)
  const wide = useStore($widthClass) === 'wide'
  const sheet = useStore($openSheet)
  const pageHasSidebar = useStore($pageHasSidebar)
  const copy = t.simplifiedAgents

  return (area, active) => {
    if (!active) {
      return null
    }

    if (area === 'chat') {
      const shown = wide ? sidebarOpen : sheet === 'sessions'

      return { shown, toggle: toggleSidebarOpen, label: shown ? copy.hideChats : copy.showChats }
    }

    if (!PANEL_AREAS.has(area) || !pageHasSidebar) {
      return null
    }

    const shown = wide ? sidebarOpen : sheet === 'page'

    return { shown, toggle: togglePageSidebar, label: shown ? copy.hidePanel : copy.showPanel }
  }
}

/** Picking a destination that has a panel shows it whole: a panel folded away elsewhere comes back
 *  with it. Below wide the panel is a sheet the band opens, so navigation leaves it closed. */
function revealPanelFor(area: string) {
  if (PANEL_AREAS.has(area) && $widthClass.get() === 'wide' && !$sidebarOpen.get()) {
    setSidebarOpen(true)
  }
}

interface RailButtonProps {
  active: boolean
  area: string
  children: ReactNode
  label: string
  metric: IaAction
  onNavigate: () => void
  panel: null | PanelToggle
}

/** One rail destination: a 40px chrome button whose tooltip names it, or, while it is open with a
 *  panel, names what a click does to that panel. The open destination keeps its selection either way. */
function RailButton({ active, area, children, label, metric, onNavigate, panel }: RailButtonProps) {
  return (
    <Tip
      label={panel ? <TipKeybindLabel actionId="view.toggleSidebar" text={panel.label} /> : label}
      placement="left-rail"
    >
      <Button
        aria-current={active ? 'page' : undefined}
        aria-expanded={panel ? panel.shown : undefined}
        aria-label={label}
        className="portal-chrome-button"
        onClick={() => {
          recordIaAction(metric)

          if (panel) {
            panel.toggle()

            return
          }

          revealPanelFor(area)
          onNavigate()
        }}
        onFocus={() => preloadRailPage(area)}
        onPointerEnter={() => preloadRailPage(area)}
        size="icon"
        variant="ghost"
      >
        {children}
      </Button>
    </Tip>
  )
}

/** The redesign's 60px global rail (Figma 906:19409): the brand badge on top, destinations under
 *  it, status and settings at the bottom. Only the redesigned shell mounts it. */
export function SimplifiedGlobalNav({ onNavigate, footer }: SimplifiedGlobalNavProps) {
  const { t } = useI18n()
  const { pathname, search } = useLocation()
  const sessionId = useStore($focusedStoredSessionId)
  const focusedSessionIsTile = useStore($focusedSessionIsTile)
  const panelFor = usePanelToggle()

  const routeView =
    pathname === SETTINGS_ROUTE && new URLSearchParams(search).get('agentProfile')
      ? 'profiles'
      : appViewForPath(pathname)

  const view = focusedSessionIsTile ? 'chat' : routeView
  const pluginPages = useSimplifiedPluginPages()
  useIaNavigationMetrics(view)

  const destinations = [
    {
      label: t.simplifiedAgents.chats,
      icon: 'comment-discussion',
      area: 'chat',
      route: sessionId ? sessionRoute(sessionId) : NEW_CHAT_ROUTE,
      active: view === 'chat'
    },
    {
      label: t.profiles.title,
      icon: 'robot',
      area: 'profiles',
      route: PROFILES_ROUTE,
      active: view === 'profiles'
    },
    {
      label: t.sidebar.nav.capabilities,
      icon: 'symbol-misc',
      area: 'capabilities',
      route: CAPABILITIES_ROUTE,
      active: view === 'capabilities'
    },
    {
      label: t.simplifiedAgents.library,
      icon: 'files',
      area: 'artifacts',
      route: ARTIFACTS_ROUTE,
      active: view === 'artifacts'
    },
    { label: t.cron.title, icon: 'watch', area: 'cron', route: CRON_ROUTE, active: view === 'cron' }
  ] as const

  const items = [
    ...destinations.map(({ icon, ...item }) => ({
      ...item,
      id: icon,
      // Figma's robot is the one glyph drawn larger than its column (906:19430).
      icon: (props: { className?: string }) =>
        icon === 'robot' ? (
          <span
            aria-hidden
            className={cn('portal-rail-robot bg-current', props.className)}
            style={{ mask: `url("${figmaRobot}") center / contain no-repeat` }}
          />
        ) : (
          <Codicon name={icon} {...props} />
        )
    })),
    ...pluginPages.map(page => ({
      ...page,
      area: 'plugin' as const,
      active: !focusedSessionIsTile && pathname === page.route
    }))
  ]

  const settingsItem: SidebarNavItem = {
    id: SETTINGS_ROUTE,
    label: t.commandCenter.settings,
    route: `${SETTINGS_ROUTE}?tab=${REDESIGN_SETTINGS_SECTIONS[0]}`,
    icon: props => <Codicon name="settings-gear" {...props} />
  }

  return (
    <nav aria-label={t.updates.appName} className="portal-rail w-(--global-nav-width) shrink-0" data-slot="global-nav">
      {/* The brand mark heads the rail on every destination; the window drags by it, as by the band. */}
      <span aria-hidden className="portal-rail-badge" data-slot="rail-badge">
        <img alt="" src={wing} />
      </span>
      <div className="portal-rail-group min-h-0 flex-1 overflow-y-auto">
        {items.map(item => {
          const button = (
            <RailButton
              active={item.active}
              area={item.area}
              key={item.id}
              label={item.label}
              metric={`ia.rail.${item.area}`}
              onNavigate={() => onNavigate(item)}
              panel={panelFor(item.area, item.active)}
            >
              <item.icon />
            </RailButton>
          )

          if (!('settingsRoute' in item) || !item.settingsRoute) {
            return button
          }

          const route = item.settingsRoute

          // The trigger listens on a layout-free wrapper: the button inside owns its own tooltip.
          return (
            <ContextMenu key={item.id}>
              <ContextMenuTrigger asChild>
                <span className="contents">{button}</span>
              </ContextMenuTrigger>
              <ContextMenuContent aria-label={item.label}>
                <ContextMenuItem onSelect={() => onNavigate({ ...item, route })}>
                  <Codicon name="settings-gear" />
                  {t.skills.plugins.settingsToggle(item.label)}
                </ContextMenuItem>
              </ContextMenuContent>
            </ContextMenu>
          )
        })}
      </div>
      <div className="portal-rail-group">
        {footer}
        {/* Settings opens its page directly: the page's own column is the menu. */}
        <RailButton
          active={view === 'settings'}
          area="settings"
          label={t.commandCenter.settings}
          metric="ia.drawer.settings"
          onNavigate={() => onNavigate(settingsItem)}
          panel={panelFor('settings', view === 'settings')}
        >
          <Codicon name="settings-gear" />
        </RailButton>
      </div>
    </nav>
  )
}
