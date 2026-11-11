import { useStore } from '@nanostores/react'

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

import { COMMAND_CENTER_ROUTE, SETTINGS_ROUTE } from '../../routes'
import type { SidebarNavItem } from '../../types'

interface SimplifiedSettingsMenuProps {
  active: boolean
  onNavigate: (item: SidebarNavItem) => void
}

export function SimplifiedSettingsMenu({ active, onNavigate }: SimplifiedSettingsMenuProps) {
  const { t } = useI18n()
  const portal = usePortalShell()
  const mode = useStore($interfaceMode)
  const n = t.settings.nav
  const s = t.settings.sections
  const settings = (tab: string) => `${SETTINGS_ROUTE}?tab=${tab}`
  const command = (section: string) => `${COMMAND_CENTER_ROUTE}?section=${section}`

  const groups = [
    {
      label: n.providerAccounts,
      icon: 'account',
      items: [{ label: n.providerAccounts, route: `${settings('providers')}&pview=accounts` }]
    },
    {
      label: n.billing,
      icon: 'credit-card',
      items: [
        { label: n.billing, route: settings('billing') },
        { label: t.commandCenter.sections.usage, route: command('usage') }
      ]
    },
    {
      label: n.gateway,
      icon: 'remote',
      items: [
        { label: n.gateway, route: settings('gateway') },
        { label: t.commandCenter.sections.system, route: command('system') }
      ]
    },
    {
      label: n.providers,
      icon: 'key',
      items: [
        { label: n.providers, route: settings('providers') },
        { label: n.apiKeys, route: settings('keys') }
      ]
    },
    {
      label: s.appearance,
      icon: 'paintcan',
      items: [
        { label: s.appearance, route: settings('config:appearance') },
        { label: n.notifications, route: settings('notifications') }
      ]
    },
    { label: s.voice, icon: 'mic', items: [{ label: s.voice, route: settings('config:voice') }] },
    { label: n.keybinds, icon: 'keyboard', items: [{ label: n.keybinds, route: settings('keybinds') }] },
    {
      label: s.safety,
      icon: 'shield',
      items: [
        { label: s.safety, route: settings('config:safety') },
        { label: n.vault, route: settings('vault') }
      ]
    },
    {
      label: n.about,
      icon: 'info',
      items: [
        { label: n.about, route: settings('about') },
        { label: s.advanced, route: settings('config:advanced') }
      ]
    }
  ] as const

  function open(route: string, label: string, icon: (typeof IA_ACTIONS.drawer)[number] | 'settings-gear') {
    recordIaAction(`ia.drawer.${icon === 'settings-gear' ? 'settings' : icon}`, 'menu')
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
            size="icon"
            variant="ghost"
          >
            <Codicon name="settings-gear" />
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="end" className="w-56" side="right">
        <DropdownMenuItem onSelect={() => open(SETTINGS_ROUTE, t.commandCenter.settings, 'settings-gear')}>
          <Codicon name="settings-gear" />
          {t.commandCenter.settings}
        </DropdownMenuItem>
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
        <DropdownMenuSeparator />
        {groups.map(group =>
          group.items.length === 1 ? (
            <DropdownMenuItem key={group.label} onSelect={() => open(group.items[0].route, group.label, group.icon)}>
              <Codicon name={group.icon} />
              {group.label}
            </DropdownMenuItem>
          ) : (
            <DropdownMenuSub key={group.label}>
              <DropdownMenuSubTrigger>
                <Codicon name={group.icon} />
                {group.label}
              </DropdownMenuSubTrigger>
              <DropdownMenuSubContent>
                {group.items.map(item => (
                  <DropdownMenuItem key={item.route} onSelect={() => open(item.route, item.label, group.icon)}>
                    {item.label}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuSubContent>
            </DropdownMenuSub>
          )
        )}
        <DropdownMenuSeparator />
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
                open(`${settings('config:appearance')}&page=general`, t.settings.appearance.toursTitle, 'question')
              }
            >
              {t.settings.appearance.toursTitle}
            </DropdownMenuItem>
            <DropdownMenuItem
              onSelect={() => open(command('maintenance'), t.commandCenter.sections.maintenance, 'tools')}
            >
              {t.commandCenter.sections.maintenance}
            </DropdownMenuItem>
          </DropdownMenuSubContent>
        </DropdownMenuSub>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
