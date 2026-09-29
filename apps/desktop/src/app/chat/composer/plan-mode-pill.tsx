import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip, TipKeybindLabel } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $planMode, togglePlanMode } from '@/store/plan-mode'

import { ACTIVE_ICON_BTN } from './control-classes'

const PILL = cn(
  'h-(--composer-control-size) shrink-0 gap-1 rounded-md px-2 text-xs font-normal',
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)

/** Plan-mode toggle: while on, each new turn is sent as `/plan <text>` (see store/plan-mode). */
export function PlanModePill({ disabled }: { disabled: boolean }) {
  const c = useI18n().t.composer
  const on = useStore($planMode)

  return (
    <Tip
      label={<TipKeybindLabel actionId="composer.planMode" text={on ? c.planModeOnHint : c.planModeOffHint} />}
      placement="control"
    >
      <Button
        aria-label={c.planMode}
        aria-pressed={on}
        className={cn(PILL, on && ACTIVE_ICON_BTN)}
        data-testid="plan-mode-pill"
        disabled={disabled}
        onClick={togglePlanMode}
        type="button"
        variant="ghost"
      >
        <Codicon name="checklist" size="0.85rem" />
        <span>{c.planMode}</span>
      </Button>
    </Tip>
  )
}
