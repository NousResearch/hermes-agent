import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'
import { useLocation } from 'react-router'

import { usePortalRedesign } from '@/app/redesign/use-portal-redesign'
import figmaRobot from '@/assets/figma-agent-robot.svg'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from '@/components/ui/context-menu'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { recordIaAction } from '@/store/ia-metrics'
import { $focusedSessionIsTile, $focusedStoredSessionId } from '@/store/session-focus'

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
import type { SidebarNavItem } from '../../types'

import { SimplifiedSettingsMenu } from './simplified-settings-menu'
import { useIaNavigationMetrics } from './use-ia-navigation-metrics'
import { useSimplifiedPluginPages } from './use-simplified-plugin-pages'

export function SimplifiedGlobalNav({ onNavigate, footer }: { onNavigate: (item: SidebarNavItem) => void; footer?: ReactNode }) {
  const portal = usePortalRedesign()
  const { t } = useI18n()
  const { pathname, search } = useLocation()
  const sessionId = useStore($focusedStoredSessionId)
  const focusedSessionIsTile = useStore($focusedSessionIsTile)
  const routeView = pathname === SETTINGS_ROUTE && new URLSearchParams(search).get('agentProfile') ? 'profiles' : appViewForPath(pathname)
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
      icon: (props: { className?: string }) =>
        portal && icon === 'robot'
          ? (
              <span
                aria-hidden
                className={cn('size-5 bg-current', props.className)}
                style={{ mask: `url("${figmaRobot}") center / contain no-repeat` }}
              />
            )
          : <Codicon name={icon} {...props} />
    })),
    ...pluginPages.map(page => ({
      ...page,
      area: 'plugin' as const,
      active: !focusedSessionIsTile && pathname === page.route
    }))
  ]

  return (
    <nav
      aria-label={t.updates.appName}
      className="mt-(--titlebar-height) flex min-h-0 w-(--global-nav-width) shrink-0 flex-col items-center gap-1 border-r border-(--sidebar-edge-border) pb-2 pt-2"
      data-slot="global-nav"
    >
      <div className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto">
        {items.map(item => {
          const button = (
            <Button
              aria-current={item.active ? 'page' : undefined}
              aria-label={item.label}
              className={cn(item.active && 'bg-(--chrome-action-hover) text-(--ui-text-primary)')}
              onClick={() => {
                recordIaAction(`ia.rail.${item.area}`)
                onNavigate(item)
              }}
              size="icon"
              variant="ghost"
            >
              <item.icon />
            </Button>
          )

          if (!('settingsRoute' in item) || !item.settingsRoute) {
            return (
              <Tip key={item.id} label={item.label}>
                {button}
              </Tip>
            )
          }

          const route = item.settingsRoute

          return (
            <ContextMenu key={item.id}>
              <Tip label={item.label}>
                <ContextMenuTrigger asChild>{button}</ContextMenuTrigger>
              </Tip>
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
      {portal && footer}
      <SimplifiedSettingsMenu active={view === 'settings'} onNavigate={onNavigate} />
    </nav>
  )
}
