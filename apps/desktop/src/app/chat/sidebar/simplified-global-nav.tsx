import { useStore } from '@nanostores/react'
import { useLocation } from 'react-router'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $focusedSessionIsTile, $focusedStoredSessionId } from '@/store/session-focus'

import {
  appViewForPath,
  ARTIFACTS_ROUTE,
  CAPABILITIES_ROUTE,
  COMMAND_CENTER_ROUTE,
  CRON_ROUTE,
  NEW_CHAT_ROUTE,
  PROFILES_ROUTE,
  sessionRoute
} from '../../routes'
import type { SidebarNavItem } from '../../types'

import { SimplifiedSettingsMenu } from './simplified-settings-menu'
import { useSimplifiedPluginPages } from './use-simplified-plugin-pages'

export function SimplifiedGlobalNav({ onNavigate }: { onNavigate: (item: SidebarNavItem) => void }) {
  const { t } = useI18n()
  const { pathname, search } = useLocation()
  const sessionId = useStore($focusedStoredSessionId)
  const tab = new URLSearchParams(search).get('tab')
  const focusedSessionIsTile = useStore($focusedSessionIsTile)
  const view = focusedSessionIsTile ? 'chat' : appViewForPath(pathname)
  const pluginPages = useSimplifiedPluginPages()
  const pluginPageActive = !focusedSessionIsTile && pluginPages.some(page => page.route === pathname)

  const destinations = [
    {
      label: t.simplifiedAgents.chats,
      icon: 'comment-discussion',
      route: sessionId ? sessionRoute(sessionId) : NEW_CHAT_ROUTE,
      active: view === 'chat'
    },
    { label: t.simplifiedAgents.settings, icon: 'robot', route: PROFILES_ROUTE, active: view === 'profiles' },
    {
      label: t.skills.tabSkills,
      icon: 'book',
      route: `${CAPABILITIES_ROUTE}?tab=skills`,
      active: view === 'capabilities' && (!tab || tab === 'skills')
    },
    {
      label: t.sidebar.nav.capabilities,
      icon: 'symbol-misc',
      active: (view === 'capabilities' && Boolean(tab && tab !== 'skills')) || pluginPageActive,
      items: [
        { label: t.connectorsPage.title, route: `${CAPABILITIES_ROUTE}?tab=connectors`, icon: 'server' },
        { label: t.skills.tabPlugins, route: `${CAPABILITIES_ROUTE}?tab=plugins`, icon: 'plug' },
        { label: t.skills.tabToolsets, route: `${CAPABILITIES_ROUTE}?tab=toolsets`, icon: 'tools' },
        {
          label: t.commandCenter.sections.maintenance,
          route: `${COMMAND_CENTER_ROUTE}?section=maintenance`,
          icon: 'wrench'
        }
      ]
    },
    { label: t.simplifiedAgents.library, icon: 'files', route: ARTIFACTS_ROUTE, active: view === 'artifacts' },
    { label: t.simplifiedAgents.routines, icon: 'watch', route: CRON_ROUTE, active: view === 'cron' }
  ]

  function open(route: string, label: string, icon: string) {
    onNavigate({ id: route, label, route, icon: props => <Codicon name={icon} {...props} /> })
  }

  return (
    <nav
      aria-label={t.updates.appName}
      className="flex w-11 shrink-0 flex-col items-center gap-1 border-r border-(--sidebar-edge-border) pb-2 pt-[calc(var(--titlebar-height)+0.5rem)]"
    >
      <div className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto">
        {destinations.map(item => {
          const button = (
            <Button
              aria-current={item.active ? 'page' : undefined}
              aria-label={item.label}
              className={cn(item.active && 'bg-(--chrome-action-hover) text-(--ui-text-primary)')}
              onClick={item.route ? () => open(item.route, item.label, item.icon) : undefined}
              size="icon"
              variant="ghost"
            >
              <Codicon name={item.icon} />
            </Button>
          )

          return item.items ? (
            <DropdownMenu key={item.label}>
              <Tip label={item.label}>
                <DropdownMenuTrigger asChild>{button}</DropdownMenuTrigger>
              </Tip>
              <DropdownMenuContent align="start" side="right">
                {item.items.map(child => (
                  <DropdownMenuItem key={child.route} onSelect={() => open(child.route, child.label, child.icon)}>
                    <Codicon name={child.icon} />
                    {child.label}
                  </DropdownMenuItem>
                ))}
                {pluginPages.length > 0 && <DropdownMenuSeparator />}
                {pluginPages.map(page => (
                  <DropdownMenuItem
                    aria-current={!focusedSessionIsTile && pathname === page.route ? 'page' : undefined}
                    key={page.id}
                    onSelect={() => onNavigate(page)}
                  >
                    <page.icon />
                    {page.label}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <Tip key={item.label} label={item.label}>
              {button}
            </Tip>
          )
        })}
      </div>
      <SimplifiedSettingsMenu onNavigate={onNavigate} />
    </nav>
  )
}
