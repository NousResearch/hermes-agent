import { useStore } from '@nanostores/react'
import { useLocation } from 'react-router'

import { SidebarPanelLabel } from '@/app/shell/sidebar-label'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { FadeScroll } from '@/components/ui/fade-scroll'
import { RowButton } from '@/components/ui/row-button'
import { Tip } from '@/components/ui/tooltip'
import { useContributions } from '@/contrib/react/use-contributions'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $interfaceMode, shownInMode } from '@/store/interface-mode'
import { $focusedSessionIsTile } from '@/store/session-focus'
import { applySidebarNavPrefs, SIDEBAR_NAV_PREFS_AREA } from '@/store/sidebar-nav'

import { CAPABILITIES_ROUTE, SIDEBAR_NAV_AREA, type SidebarNavContribution } from '../../routes'
import type { SidebarNavItem } from '../../types'

export function SimplifiedPluginPages({ onNavigate }: { onNavigate: (item: SidebarNavItem) => void }) {
  const { t } = useI18n()
  const { pathname } = useLocation()
  const contributions = useContributions(SIDEBAR_NAV_AREA)
  const preferences = useContributions(SIDEBAR_NAV_PREFS_AREA)
  const mode = useStore($interfaceMode)
  const focusedSessionIsTile = useStore($focusedSessionIsTile)

  const pages = applySidebarNavPrefs(
    contributions.flatMap(contribution => {
      const data = contribution.data as Partial<SidebarNavContribution> | undefined

      if (!data?.path?.startsWith('/') || !data.label || !shownInMode(mode)(data)) {
        return []
      }

      return [
        {
          id: contribution.id,
          label: data.label,
          route: data.path,
          plugin: contribution.source?.startsWith('plugin:') ? contribution.source.slice(7) : undefined,
          icon: (props: { className?: string }) => <Codicon name={data.codicon || 'plug'} {...props} />
        }
      ]
    }),
    preferences
  )

  if (!pages.length) {
    return null
  }

  return (
    <section aria-label={t.skills.tabPlugins} className="flex max-h-[25%] shrink-0 flex-col">
      <SidebarPanelLabel className="shrink-0 px-2 py-2 [&>span:first-child]:mx-1">
        {t.skills.tabPlugins}
      </SidebarPanelLabel>
      <FadeScroll className="min-h-0" fade="min(2.5rem, 25%)" maxHeight="none">
        {pages.map(item => {
          const active = !focusedSessionIsTile && pathname === item.route
          const settingsLabel = `${t.commandCenter.settings}: ${item.label}`

          return (
            <div className="flex items-center gap-1" key={item.id}>
              <RowButton
                aria-current={active ? 'page' : undefined}
                className={cn(
                  'flex min-w-0 flex-1 items-center gap-2 rounded-sm px-2 py-2 text-left text-xs hover:bg-(--chrome-action-hover) focus-visible:outline-2 focus-visible:outline-ring',
                  active && 'bg-(--chrome-action-hover)'
                )}
                onClick={() => onNavigate(item)}
              >
                <item.icon />
                <span className="truncate">{item.label}</span>
              </RowButton>
              {item.plugin && (
                <Tip label={settingsLabel}>
                  <Button
                    aria-label={settingsLabel}
                    onClick={() =>
                      onNavigate({
                        ...item,
                        route: `${CAPABILITIES_ROUTE}?tab=plugins&plugin=${encodeURIComponent(item.plugin!)}`
                      })
                    }
                    size="icon-xs"
                    variant="ghost"
                  >
                    <Codicon name="settings-gear" size={12} />
                  </Button>
                </Tip>
              )}
            </div>
          )
        })}
      </FadeScroll>
    </section>
  )
}
