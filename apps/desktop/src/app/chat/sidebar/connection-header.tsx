import { useNavigate } from 'react-router'

import { SETTINGS_ROUTE } from '@/app/routes'
import { SidebarGroup } from '@/components/ui/sidebar'

import { ConnectionSwitcher } from './connection-switcher'

export function ConnectionHeader() {
  const navigate = useNavigate()

  return (
    <SidebarGroup className="shrink-0 px-2.5">
      <ConnectionSwitcher onConnect={() => navigate(`${SETTINGS_ROUTE}?tab=connections`)} />
    </SidebarGroup>
  )
}
