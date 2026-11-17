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
import { cn } from '@/lib/utils'

import { openConnectorsAdmin } from './data/portal'

export interface ConnectorDialogMenuProps {
  onDisconnect?: () => void
  onReconnect?: () => void
  onRefreshTools: () => void
}

export function ConnectorDialogMenu({ onDisconnect, onReconnect, onRefreshTools }: ConnectorDialogMenuProps) {
  const { t } = useI18n()
  const copy = t.connectorsPage

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button aria-label={copy.dialog.moreActions} size="icon-xs" variant="ghost">
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
        <DropdownMenuItem className={dropdownMenuRow} onSelect={() => void openConnectorsAdmin()}>
          {copy.dialog.orgLink}
        </DropdownMenuItem>
        {onDisconnect ? (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className={cn(dropdownMenuRow, 'text-destructive focus:text-destructive')}
              onSelect={onDisconnect}
            >
              {copy.dialog.disconnect}
            </DropdownMenuItem>
          </>
        ) : null}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
