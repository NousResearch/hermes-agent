import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { AlertTriangle, Check, ChevronDown } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { setSessionYolo } from '@/lib/yolo-session'
import { $activeSessionId, $yoloActive, setYoloActive } from '@/store/session'

// Codex-style approval-mode selector in the composer (China-first batch #4).
// Two tiers that map onto the runtime's existing binary approval toggle:
//   review (替我审批) → approvals bypass OFF (`yolo=0`): the agent runs, only
//                       detected-risk (dangerous) operations ask for approval.
//   full   (完全访问) → approvals bypass ON  (`yolo=1`): unrestricted.
// The third Codex tier (请求批准 / always-gate file+internet) needs category
// gating the runtime doesn't have yet — deferred / scheduled, not shown here.
type ApprovalMode = 'review' | 'full'

const PILL = cn(
  'h-(--composer-control-size) shrink-0 gap-1 rounded-md px-2 text-xs font-normal',
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)

export function ApprovalPill({ disabled }: { disabled: boolean }) {
  const t = useI18n().t.composer.approvalMode
  const { requestGateway } = useGatewayRequest()
  const yoloActive = useStore($yoloActive)
  const sessionId = useStore($activeSessionId)
  const [open, setOpen] = useState(false)

  const mode: ApprovalMode = yoloActive ? 'full' : 'review'

  async function select(next: ApprovalMode) {
    setOpen(false)

    if (next === mode) {
      return
    }

    const enabled = next === 'full'

    // Session-scoped when a conversation exists (matches the status-bar zap and
    // use-session-actions, which re-applies $yoloActive to new sessions); before
    // the first session, flip the local flag so the next session inherits it.
    if (sessionId) {
      await setSessionYolo(requestGateway, sessionId, enabled).catch(() => setYoloActive(enabled))
    } else {
      setYoloActive(enabled)
    }
  }

  const TriggerIcon = mode === 'full' ? AlertTriangle : Check

  return (
    <DropdownMenu onOpenChange={setOpen} open={open}>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={t.label}
          className={cn(PILL, mode === 'full' && 'text-amber-500 hover:text-amber-500')}
          disabled={disabled}
          title={t.label}
          type="button"
          variant="ghost"
        >
          <TriggerIcon className="size-3.5 shrink-0" />
          <span className="truncate">{t[mode].label}</span>
          <ChevronDown className="size-2.5 shrink-0 opacity-50" />
        </Button>
      </DropdownMenuTrigger>
      {/* Wide enough that each tier's one-line description never wraps (Codex
          keeps these single-line). */}
      <DropdownMenuContent align="start" className="w-96" side="top" sideOffset={8}>
        <DropdownMenuRadioGroup onValueChange={value => void select(value as ApprovalMode)} value={mode}>
          {(['review', 'full'] as const).map(value => {
            const Icon = value === 'full' ? AlertTriangle : Check

            return (
              <DropdownMenuRadioItem className="items-start gap-2 py-1.5" key={value} value={value}>
                <Icon className="mt-0.5 size-4 shrink-0 text-(--ui-text-tertiary)" />
                <span className="flex flex-col gap-0.5">
                  <span className="text-foreground">{t[value].label}</span>
                  <span className="text-(--ui-text-tertiary)">{t[value].desc}</span>
                </span>
              </DropdownMenuRadioItem>
            )
          })}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
