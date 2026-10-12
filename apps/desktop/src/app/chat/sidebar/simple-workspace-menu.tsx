import { useStore } from '@nanostores/react'

import { toggleLayoutEditMode } from '@/components/pane-shell/edit-mode'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { KbdCombo } from '@/components/ui/kbd'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { $bindings, bindingsFor } from '@/store/keybinds'

import { SETTINGS_ROUTE } from '../../routes'
import type { SidebarNavItem } from '../../types'

interface SimpleWorkspaceMenuProps {
  activeRoute?: string
  items: SidebarNavItem[]
  onNavigate: (item: SidebarNavItem) => void
}

/** Secondary destinations retain their existing route and plugin actions. */
export function SimpleWorkspaceMenu({ activeRoute, items, onNavigate }: SimpleWorkspaceMenuProps) {
  const { t } = useI18n()
  const bindings = useStore($bindings)
  const activeItem = items.find(item => item.route === activeRoute)
  const activeLabel = activeItem ? (t.sidebar.nav[activeItem.id] ?? activeItem.label) : null

  return (
    <div className="flex shrink-0 items-center justify-between gap-1 py-2 [-webkit-app-region:no-drag]">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button className="min-w-0 shrink" size="sm" variant="ghost">
            <Codicon name="ellipsis" />
            <span className="truncate">
              {activeLabel ? `${t.simpleWorkspace.more} · ${activeLabel}` : t.simpleWorkspace.more}
            </span>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" side="top">
          {items.map(item => {
            const combo = item.keybindActionId ? bindingsFor(item.keybindActionId, bindings)[0] : null

            return (
              <DropdownMenuItem
                aria-current={item === activeItem ? 'page' : undefined}
                key={item.id}
                onSelect={() => onNavigate(item)}
              >
                <item.icon />
                {t.sidebar.nav[item.id] ?? item.label}
                {combo && <KbdCombo className="ml-auto" combo={combo} size="sm" variant="ghost" />}
                {item === activeItem && <Codicon aria-hidden="true" name="check" />}
              </DropdownMenuItem>
            )
          })}
        </DropdownMenuContent>
      </DropdownMenu>
      <Tip label={t.titlebar.layoutEditor}>
        <Button aria-label={t.titlebar.layoutEditor} onClick={toggleLayoutEditMode} size="icon-xs" variant="ghost">
          <Codicon name="layout" />
        </Button>
      </Tip>
      <Button
        onClick={() =>
          onNavigate({
            id: 'settings',
            label: t.commandCenter.settings,
            route: SETTINGS_ROUTE,
            icon: props => <Codicon name="settings-gear" {...props} />
          })
        }
        size="sm"
        variant="ghost"
      >
        <Codicon name="settings-gear" />
        {t.commandCenter.settings}
      </Button>
    </div>
  )
}
