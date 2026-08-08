import {
  ApprovalModeGlyph,
  ApprovalModeMenuContent,
  approvalModeToneClass,
  useApprovalModeMenu
} from '@/app/shell/approval-mode-menu'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Tip } from '@/components/ui/tooltip'
import { ChevronDown } from '@/lib/icons'
import { cn } from '@/lib/utils'
import type { ApprovalModeRequester } from '@/store/approval-mode'

/** A compact profile-scoped approvals selector that lives beside the composer
 * attachment menu. The popover opens upward so its options stay visually
 * attached to the chat bar instead of covering the conversation. */
export function ComposerApprovalMode({
  disabled = false,
  profile,
  requestGateway
}: {
  disabled?: boolean
  profile: string
  requestGateway: ApprovalModeRequester
}) {
  // The persistent status-bar control owns the one profile sync. Every
  // ChatBar (including split-session tiles) renders this control, so syncing
  // here would fan one config read out across every open pane.
  const menu = useApprovalModeMenu(profile, requestGateway, { sync: false })

  return (
    <DropdownMenu>
      <Tip label={menu.ariaLabel} side="top">
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={menu.ariaLabel}
            className={cn(
              'h-7 gap-1 rounded-[var(--radius-sm)] px-1.5 py-0 text-[0.6875rem] font-medium text-(--ui-text-secondary)',
              'hover:bg-(--chrome-action-hover) hover:text-foreground data-[state=open]:bg-(--chrome-action-hover) data-[state=open]:text-foreground'
            )}
            data-approval-mode={menu.mode}
            disabled={disabled}
            motion="none"
            size="inline"
            type="button"
            variant="ghost"
          >
            <ApprovalModeGlyph className="size-3" mode={menu.mode} />
            <span className={cn('max-w-24 truncate', approvalModeToneClass[menu.mode])}>{menu.labels[menu.mode]}</span>
            <ChevronDown className="size-3 text-(--ui-text-tertiary)" />
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="start" className="w-72 p-1" side="top" sideOffset={8}>
        <ApprovalModeMenuContent menu={menu} />
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
