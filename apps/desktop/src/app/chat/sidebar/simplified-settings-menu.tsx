import { useStore } from '@nanostores/react'

import { useRailMenuCollisionPadding } from '@/app/redesign/portal-status-menu'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { openExternalLink } from '@/lib/external-link'
import { recordIaAction } from '@/store/ia-metrics'
import type { IA_ACTIONS } from '@/store/ia-metrics-contract'
import { $interfaceMode, INTERFACE_MODES, setInterfaceMode } from '@/store/interface-mode'
import { usePortalShell } from '@/store/shell-variant'

import { preloadRailPage } from '../../page-views'
import { COMMAND_CENTER_ROUTE, SETTINGS_ROUTE } from '../../routes'
import {
  REDESIGN_SETTINGS_SECTIONS,
  redesignSettingsLabels,
  type RedesignSettingsSection
} from '../../settings/redesign-navigation'
import { redesignNavigationCopy } from '../../settings/redesign-navigation-copy'
import type { SidebarNavItem } from '../../types'

type DrawerMetric = (typeof IA_ACTIONS.drawer)[number]

interface MenuEntry {
  icon: string
  /** The closed IA counter (ia-metrics-contract.ts); a section without its own counts with the
   *  group it sat in before the menu followed the page. */
  metric: DrawerMetric
}

const SECTION_MENU: Record<RedesignSettingsSection, MenuEntry> = {
  billing: { icon: 'credit-card', metric: 'credit-card' },
  'config:appearance': { icon: 'paintcan', metric: 'paintcan' },
  notifications: { icon: 'bell', metric: 'paintcan' },
  keybinds: { icon: 'keyboard', metric: 'keyboard' },
  gateway: { icon: 'remote', metric: 'remote' },
  sessions: { icon: 'archive', metric: 'settings' },
  about: { icon: 'info', metric: 'info' },
  'config:advanced': { icon: 'settings', metric: 'info' }
}

interface SimplifiedSettingsMenuProps {
  active: boolean
  onNavigate: (item: SidebarNavItem) => void
}

export function SimplifiedSettingsMenu({ active, onNavigate }: SimplifiedSettingsMenuProps) {
  const { locale, t } = useI18n()
  const portal = usePortalShell()
  const mode = useStore($interfaceMode)
  const collisionPadding = useRailMenuCollisionPadding()
  const settings = (tab: string) => `${SETTINGS_ROUTE}?tab=${tab}`
  const command = (section: string) => `${COMMAND_CENTER_ROUTE}?section=${section}`
  const labels = redesignSettingsLabels(t, redesignNavigationCopy(locale))

  function open(route: string, label: string, { icon, metric }: MenuEntry) {
    recordIaAction(`ia.drawer.${metric}`, 'menu')
    onNavigate({ id: route, label, route, icon: props => <Codicon name={icon} {...props} /> })
  }

  return (
    <DropdownMenu
      onOpenChange={open => {
        if (open) {
          recordIaAction('ia.drawer.opened')
        }
      }}
    >
      <Tip label={t.commandCenter.settings} placement="left-rail">
        <DropdownMenuTrigger asChild>
          <Button
            aria-current={active ? 'page' : undefined}
            aria-label={t.commandCenter.settings}
            className="portal-chrome-button"
            onFocus={() => preloadRailPage('settings')}
            onPointerEnter={() => preloadRailPage('settings')}
            size="icon"
            variant="ghost"
          >
            <Codicon name="settings-gear" />
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent
        align="end"
        className="portal-rail-popover w-56"
        collisionPadding={collisionPadding}
        side="right"
        sideOffset={0}
      >
        {/* The Settings page's own destinations, in its order and under its names. */}
        {REDESIGN_SETTINGS_SECTIONS.map(section => (
          <DropdownMenuItem
            key={section}
            onSelect={() => open(settings(section), labels[section], SECTION_MENU[section])}
          >
            <Codicon name={SECTION_MENU[section].icon} />
            {labels[section]}
          </DropdownMenuItem>
        ))}
        <DropdownMenuSeparator />
        {portal && (
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>
              <Codicon name="layout" />
              {t.interfaceMode.title}
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              <DropdownMenuRadioGroup value={mode}>
                {INTERFACE_MODES.map(id => (
                  <DropdownMenuRadioItem key={id} onSelect={() => setInterfaceMode(id)} value={id}>
                    {t.interfaceMode[id].label}
                  </DropdownMenuRadioItem>
                ))}
              </DropdownMenuRadioGroup>
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        )}
        <DropdownMenuSub>
          <DropdownMenuSubTrigger>
            <Codicon name="question" />
            {t.catalog.documentation}
          </DropdownMenuSubTrigger>
          <DropdownMenuSubContent>
            <DropdownMenuItem
              onSelect={() => {
                recordIaAction('ia.drawer.documentation', 'menu')
                openExternalLink('https://hermes-agent.nousresearch.com/docs/')
              }}
            >
              {t.catalog.documentation}
            </DropdownMenuItem>
            <DropdownMenuItem
              onSelect={() =>
                open(`${settings('config:appearance')}&page=general`, t.settings.appearance.toursTitle, {
                  icon: 'question',
                  metric: 'question'
                })
              }
            >
              {t.settings.appearance.toursTitle}
            </DropdownMenuItem>
            <DropdownMenuItem
              onSelect={() =>
                open(command('maintenance'), t.commandCenter.sections.maintenance, { icon: 'tools', metric: 'tools' })
              }
            >
              {t.commandCenter.sections.maintenance}
            </DropdownMenuItem>
          </DropdownMenuSubContent>
        </DropdownMenuSub>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
