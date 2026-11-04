import { BrandMark } from '@/components/brand-mark'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'

import { CAPABILITIES_ROUTE } from '../../routes'
import type { SidebarNavItem } from '../../types'

export function SimplifiedGlobalMenu({ onNavigate }: { onNavigate: (item: SidebarNavItem) => void }) {
  const { t } = useI18n()

  const destinations = [
    { id: 'skills', label: t.skills.tabSkills, codicon: 'book' },
    { id: 'plugins', label: t.skills.tabPlugins, codicon: 'plug' },
    { id: 'connectors', label: t.connectorsPage.title, codicon: 'server' }
  ]

  function open(id: string, label: string, route: string, codicon: string) {
    onNavigate({ id, label, route, icon: props => <Codicon name={codicon} {...props} /> })
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button className="w-full justify-start" variant="ghost">
          <BrandMark className="size-6" />
          <span className="flex-1 text-left">{t.updates.appName}</span>
          <Codicon name="chevron-down" size={12} />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-56">
        {destinations.map(item => (
          <DropdownMenuItem
            key={item.id}
            onSelect={() => open(item.id, item.label, `${CAPABILITIES_ROUTE}?tab=${item.id}`, item.codicon)}
          >
            <Codicon name={item.codicon} />
            {item.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
