import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  dropdownMenuRow,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'

export interface ConnectorDialogMenuProps {
  onRefreshTools: () => void
}

export function ConnectorDialogMenu({ onRefreshTools }: ConnectorDialogMenuProps) {
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
        <DropdownMenuItem className={dropdownMenuRow} onSelect={onRefreshTools}>
          {copy.dialog.menuRefreshTools}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
