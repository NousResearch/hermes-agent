import { useStore } from '@nanostores/react'
import { useLocation } from 'react-router'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
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
  sessionRoute
} from '../../routes'
import type { SidebarNavItem } from '../../types'

import { SimplifiedSettingsMenu } from './simplified-settings-menu'
import { useIaNavigationMetrics } from './use-ia-navigation-metrics'
import { useSimplifiedPluginPages } from './use-simplified-plugin-pages'

export function SimplifiedGlobalNav({ onNavigate }: { onNavigate: (item: SidebarNavItem) => void }) {
  const { t } = useI18n()
  const { pathname } = useLocation()
  const sessionId = useStore($focusedStoredSessionId)
  const focusedSessionIsTile = useStore($focusedSessionIsTile)
  const view = focusedSessionIsTile ? 'chat' : appViewForPath(pathname)
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
      label: t.simplifiedAgents.settings,
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
    { label: t.simplifiedAgents.routines, icon: 'watch', area: 'cron', route: CRON_ROUTE, active: view === 'cron' }
  ] as const

  const items = [
    ...destinations.map(({ icon, ...item }) => ({
      ...item,
      id: icon,
      icon: (props: { className?: string }) => <Codicon name={icon} {...props} />
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
      className="flex w-11 shrink-0 flex-col items-center gap-1 border-r border-(--sidebar-edge-border) pb-2 pt-[calc(var(--titlebar-height)+0.5rem)]"
    >
      <div className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto">
        {items.map(item => (
          <Tip key={item.id} label={item.label}>
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
          </Tip>
        ))}
      </div>
      <SimplifiedSettingsMenu onNavigate={onNavigate} />
    </nav>
  )
}
