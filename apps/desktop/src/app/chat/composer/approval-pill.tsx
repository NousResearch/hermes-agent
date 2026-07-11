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
import { AlertTriangle, Check, ChevronDown, Sparkles } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { applyApprovalMode, setSessionYolo } from '@/lib/yolo-session'
import { $activeSessionId, $approvalMode, $yoloActive, setYoloActive } from '@/store/session'

// Three-tier approval selector in the composer (hc-514):
//   manual → GLOBAL approvals.mode="manual": gate ONLY commands the runtime
//            flags as dangerous (tools/approval.py detect_dangerous_command).
//            Honest copy — this is NOT a Codex-style "ask on every external
//            file write / internet use"; that gate doesn't exist yet (hc-514 b).
//   smart  → GLOBAL approvals.mode="smart": an assistant LLM weighs risk, asks.
//   full   → SESSION-scoped yolo override (the pre-hc-514 mechanism): temporary,
//            this chat only, gone when the session ends — the next session falls
//            back to the global gating tier. The pill NEVER writes the
//            persistent global approvals.mode to "off": a persistent
//            unrestricted default (affecting CLI / TUI / cron too) would be a
//            security regression over the old session-level toggle.
// The asymmetry is deliberate: the gating tiers are restrictive, so persisting
// them globally is safe (approvals.mode has no per-session form anyway); the
// permissive tier stays session-local. A legacy config where an older build
// left approvals.mode=off still renders as "full", and picking manual/smart
// from there re-homes the global mode to the chosen tier.
type ApprovalTier = 'full' | 'manual' | 'smart'

const TIERS = ['manual', 'smart', 'full'] as const

const TIER_ICON = { full: AlertTriangle, manual: Check, smart: Sparkles } as const

const PILL = cn(
  'h-(--composer-control-size) shrink-0 gap-1 rounded-md px-2 text-xs font-normal',
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)

export function ApprovalPill({ disabled }: { disabled: boolean }) {
  const t = useI18n().t.composer.approvalMode
  const { requestGateway } = useGatewayRequest()
  const yoloActive = useStore($yoloActive)
  const approvalMode = useStore($approvalMode)
  const sessionId = useStore($activeSessionId)
  const [open, setOpen] = useState(false)

  // "full" whenever approvals are effectively bypassed — either the global mode
  // is off, or a per-session yolo override (status-bar zap) is armed for this
  // chat. Otherwise reflect the global gating mode.
  const tier: ApprovalTier =
    yoloActive || approvalMode === 'off' ? 'full' : approvalMode === 'smart' ? 'smart' : 'manual'

  async function select(next: ApprovalTier) {
    setOpen(false)

    if (next === tier) {
      return
    }

    if (next === 'full') {
      // Session-scoped temporary override only — never the persistent global
      // mode. Before the first session exists, arm the local flag so the next
      // session inherits it (use-session-actions re-applies it at creation),
      // matching the pre-hc-514 behavior.
      if (sessionId) {
        await setSessionYolo(requestGateway, sessionId, true).catch(() => setYoloActive(true))
      } else {
        setYoloActive(true)
      }

      return
    }

    // Switching to a gating tier: clear any per-session yolo override first so
    // the global gate actually applies to this chat instead of the pick
    // snapping straight back to "full".
    if (sessionId && yoloActive) {
      await setSessionYolo(requestGateway, sessionId, false).catch(() => setYoloActive(false))
    } else if (yoloActive) {
      setYoloActive(false)
    }

    // Persist the picked gating tier globally. If a legacy build left
    // approvals.mode=off, this same write re-homes the global default.
    await applyApprovalMode(requestGateway, next === 'smart' ? 'smart' : 'manual')
  }

  const TriggerIcon = TIER_ICON[tier]

  return (
    <DropdownMenu onOpenChange={setOpen} open={open}>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={t.label}
          className={cn(PILL, tier === 'full' && 'text-amber-500 hover:text-amber-500')}
          disabled={disabled}
          title={t.label}
          type="button"
          variant="ghost"
        >
          <TriggerIcon className="size-3.5 shrink-0" />
          <span className="truncate">{t[tier].label}</span>
          <ChevronDown className="size-2.5 shrink-0 opacity-50" />
        </Button>
      </DropdownMenuTrigger>
      {/* Wide enough that each tier's one-line description never wraps. */}
      <DropdownMenuContent align="start" className="w-96" side="top" sideOffset={8}>
        <DropdownMenuRadioGroup onValueChange={value => void select(value as ApprovalTier)} value={tier}>
          {TIERS.map(value => {
            const Icon = TIER_ICON[value]

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
