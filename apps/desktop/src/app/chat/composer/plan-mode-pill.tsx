import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $planMode, togglePlanMode } from '@/store/plan-mode'

/**
 * Plan-mode toggle pill (Cursor-style). When on, the composer routes each
 * message through the `/plan` skill — the agent writes an actionable plan to
 * `.hermes/plans/` instead of executing. Cache-safe: it only prepends `/plan`
 * to the submitted text.
 */
export function PlanModePill({ disabled }: { disabled: boolean }) {
  const c = useI18n().t.composer
  const on = useStore($planMode)

  return (
    <Tip label={on ? c.planModeOnHint : c.planModeOffHint}>
      <Button
        aria-label={c.planMode}
        aria-pressed={on}
        className={cn(
          'h-(--composer-control-size) shrink-0 gap-1 rounded-md px-2 text-xs font-normal transition-colors',
          on
            ? 'bg-(--ui-accent,#3b82f6)/15 text-(--ui-accent,#3b82f6) hover:bg-(--ui-accent,#3b82f6)/25'
            : 'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
        )}
        disabled={disabled}
        onClick={() => togglePlanMode()}
        type="button"
        variant="ghost"
      >
        <Codicon name="checklist" size="0.85rem" />
        <span>{c.planMode}</span>
      </Button>
    </Tip>
  )
}
