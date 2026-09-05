/**
 * The visible half of directive diagnostics.
 *
 * `directive-diagnostics.ts` explains why a dropped directive must announce
 * itself; the log alone only reaches someone who already suspects a bug and
 * knows to open devtools. The user's symptom is raw `::followup{...}` sitting
 * in the transcript looking like the model emitted junk.
 *
 * This badge replaces that raw source with a quiet, one-line marker naming the
 * reason. Deliberately small and muted: a dropped panel is an app-side defect,
 * but it is not worth an alarm banner mid-conversation, and directives can
 * appear in ordinary prose (docs, this very changelog) where a loud error
 * would be wrong.
 *
 * The raw text stays reachable through the tooltip and the title attribute so
 * a bug report can still quote it verbatim.
 */

import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

/** Longest raw source echoed into the tooltip; directives can be long. */
const MAX_TOOLTIP_SOURCE = 300

export interface DirectiveDropBadgeProps {
  /** Why the directive never became a widget, in user-facing words. */
  reason: string
  /** The paragraph's raw source, for the tooltip and bug reports. */
  source: string
  className?: string
}

export function DirectiveDropBadge({ className, reason, source }: DirectiveDropBadgeProps) {
  const trimmed = source.trim()
  const quoted = trimmed.length > MAX_TOOLTIP_SOURCE ? `${trimmed.slice(0, MAX_TOOLTIP_SOURCE)}…` : trimmed

  return (
    <Tip label={quoted} side="top">
      <span
        className={cn(
          'inline-flex select-none items-center gap-1.5 rounded-md border border-dashed',
          'border-border/60 px-2 py-0.5 text-xs text-muted-foreground',
          className
        )}
        data-testid="directive-drop-badge"
        title={quoted}
      >
        <Codicon name="debug-disconnect" size="0.85em" />
        {reason}
      </span>
    </Tip>
  )
}
