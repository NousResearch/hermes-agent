// The dialog's kebab: the three things that are not the tool list.
//
// Reconnect re-mints the authorization, Refresh tools revalidates the 24 h
// cache, and the admin link leaves the app for the portal's connectors page —
// the one place an organisation's rules can actually be changed.

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  dropdownMenuRow,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { openExternalLink } from '@/lib/external-link'

import { useConnectorsAdminUrl } from './data/portal'

export interface ConnectorDialogMenuProps {
  /** Absent on a server on this Mac: there is nothing to reconnect to. */
  onReconnect?: () => void
  onRefreshTools: () => void
}

export function ConnectorDialogMenu({ onReconnect, onRefreshTools }: ConnectorDialogMenuProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage
  const adminUrl = useConnectorsAdminUrl()

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button aria-label={copy.dialog.moreActions} size="icon" variant="ghost">
          <Codicon name="ellipsis" size="0.8125rem" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {onReconnect ? (
          <DropdownMenuItem className={dropdownMenuRow} onSelect={onReconnect}>
            {copy.card.verb.reconnect}
          </DropdownMenuItem>
        ) : null}
        <DropdownMenuItem className={dropdownMenuRow} onSelect={onRefreshTools}>
          {copy.dialog.menuRefreshTools}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem className={dropdownMenuRow} onSelect={() => openExternalLink(adminUrl)}>
          {copy.dialog.orgLink}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
