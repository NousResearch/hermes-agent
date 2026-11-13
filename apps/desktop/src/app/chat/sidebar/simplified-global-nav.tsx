import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'
import { useLocation } from 'react-router'

import figmaRobot from '@/assets/figma-agent-robot.svg'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from '@/components/ui/context-menu'
import { Tip, TipKeybindLabel } from '@/components/ui/tooltip'
import { useContributions } from '@/contrib/react/use-contributions'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $cronJobs } from '@/store/cron'
import { recordIaAction } from '@/store/ia-metrics'
import type { IaAction } from '@/store/ia-metrics-contract'
import { $sidebarOpen, setSidebarOpen, toggleSidebarOpen } from '@/store/layout'
import { $focusedSessionIsTile, $focusedStoredSessionId } from '@/store/session-focus'
import { applySidebarNavPrefs, SIDEBAR_NAV_PREFS_AREA } from '@/store/sidebar-nav'

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
import { routineNeedsAttention } from '../../routines/routine-model'
import { useRoutinesCopy } from '../../routines/routines-copy'
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
  /** Something here needs the user (Routines: a failed or overdue routine), as a dot on the glyph. */
  alert?: string
  label: string
  metric: IaAction
  onNavigate: () => void
  panel: null | PanelToggle
}

/** One rail destination: a 30px chrome button whose tooltip names it, or, while it is open with a
 *  panel, names what a click does to that panel. The open destination keeps its selection either way. */
function RailButton({ active, alert, area, children, label, metric, onNavigate, panel }: RailButtonProps) {
  return (
    <Tip
      label={panel ? <TipKeybindLabel actionId="view.toggleSidebar" text={panel.label} /> : alert ? `${label} · ${alert}` : label}
      placement="left-rail"
    >
      <Button
        aria-current={active ? 'page' : undefined}
        aria-description={alert}
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
        {alert && <span aria-hidden className="portal-rail-alert" data-slot="rail-alert" />}
      </Button>
    </Tip>
  )
}

/** The Routines destination's dot (proposal point 3): how many routines need attention, from the
 *  snapshot the sidebar sync keeps for every agent. Nothing when all is well. */
function useRoutinesAlert(): string | undefined {
  const jobs = useStore($cronJobs)
  const copy = useRoutinesCopy()
  const count = jobs.filter(job => routineNeedsAttention(job)).length

  return count ? copy.railAttention(count) : undefined
}

/** The redesign's 60px global rail (rail 16292:15233): destinations from the top, status and
 *  settings at its foot. The brand moved to the chat sidebar's wordmark; Figma hides the rail's badge
 *  layer and draws the window's traffic lights in its place. Only the redesigned shell mounts it. */
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
  const navPrefs = useContributions(SIDEBAR_NAV_PREFS_AREA)
  const routinesAlert = useRoutinesAlert()
  useIaNavigationMetrics(view)

  // Chats and Agents lead the rail: the redesign's homes for what classic keeps outside its nav rows
  // (the session list, the profile switcher), so a nav preference never moves or hides them, just as
  // Settings stays at the foot. The rest are classic's nav rows by the same ids, and plugin pages.
  const anchored = [
    {
      id: 'chat',
      label: t.simplifiedAgents.chats,
      icon: 'comment-discussion',
      area: 'chat',
      route: sessionId ? sessionRoute(sessionId) : NEW_CHAT_ROUTE,
      active: view === 'chat'
    },
    {
      id: 'profiles',
      label: t.profiles.title,
      icon: 'robot',
      area: 'profiles',
      route: PROFILES_ROUTE,
      active: view === 'profiles'
    }
  ] as const

  const movable = [
    {
      id: 'capabilities',
      label: t.sidebar.nav.capabilities,
      icon: 'symbol-misc',
      area: 'capabilities',
      route: CAPABILITIES_ROUTE,
      active: view === 'capabilities'
    },
    {
      id: 'artifacts',
      label: t.simplifiedAgents.library,
      icon: 'files',
      area: 'artifacts',
      route: ARTIFACTS_ROUTE,
      active: view === 'artifacts'
    },
    // Figma's Routines glyph is a clock (16292:15264).
    { id: 'cron', label: t.cron.title, icon: 'clock', area: 'cron', route: CRON_ROUTE, active: view === 'cron' }
  ] as const

  const coreItem = ({ icon, ...item }: (typeof anchored)[number] | (typeof movable)[number]) => ({
    ...item,
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
  })

  const items = [
    ...anchored.map(coreItem),
    ...applySidebarNavPrefs(
      [
        ...movable.map(coreItem),
        ...pluginPages.map(page => ({
          ...page,
          area: 'plugin' as const,
          // A plugin page owns its route's subtree (`/board/settings` is still the board).
          active: !focusedSessionIsTile && (pathname === page.route || pathname.startsWith(`${page.route}/`))
        }))
      ],
      navPrefs
    )
  ]

  const settingsItem: SidebarNavItem = {
    id: SETTINGS_ROUTE,
    label: t.commandCenter.settings,
    route: `${SETTINGS_ROUTE}?tab=${REDESIGN_SETTINGS_SECTIONS[0]}`,
    icon: props => <Codicon name="settings-gear" {...props} />
  }

  return (
    <nav aria-label={t.updates.appName} className="portal-rail w-(--global-nav-width) shrink-0" data-slot="global-nav">
      <div className="portal-rail-group min-h-0 flex-1 overflow-y-auto">
        {items.map(item => {
          const button = (
            <RailButton
              active={item.active}
              alert={item.area === 'cron' ? routinesAlert : undefined}
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
      <div className="portal-rail-group portal-rail-foot">
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
