/**
 * Shared drag-and-drop visual language — ONE dashed sheet + ONE floating pill,
 * used by every drop affordance (the composer file/session overlay, the layout
 * zone targets) so "you can drop here" reads identically everywhere.
 */

import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

/** The sheet: a dashed region marking where a drop would land. */
export const DROP_SHEET_CLASS = 'rounded-2xl border-2 border-dashed'

/** Soft blur for the LIVE sheet only — idle outlines must not fog the app. */
export const DROP_SHEET_BLUR_CLASS = 'backdrop-blur-[2px] [-webkit-backdrop-filter:blur(2px)]'

/** The pill: icon + label floating over a sheet, naming the outcome. */
export function DropPill({ children, className, icon }: React.ComponentProps<'div'> & { icon: string }) {
  return (
    <div
      className={cn(
        'flex items-center gap-2 rounded-full border border-[color-mix(in_srgb,var(--dt-composer-ring)_45%,transparent)] bg-[color-mix(in_srgb,var(--dt-card)_92%,transparent)] px-4 py-2 text-[0.8125rem] font-medium text-foreground shadow-composer',
        className
      )}
    >
      <Codicon className="shrink-0 text-(--ui-accent)" name={icon} size="1rem" />
      {children}
    </div>
  )
}
