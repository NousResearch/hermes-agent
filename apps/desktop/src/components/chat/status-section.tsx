import { type ReactNode, useState } from 'react'

import { DisclosureCaret } from '@/components/ui/disclosure-caret'

interface StatusSectionProps {
  /** Optional right-aligned actions (text links / micro buttons). Pass
   *  `Button` with `size="micro"` + `variant="text"` or `"link"`. */
  accessory?: ReactNode
  children: ReactNode
  /** Optional inline status shown only while the group is collapsed. */
  collapsedIndicator?: ReactNode
  defaultCollapsed?: boolean
  /** Optional glyph between the caret and the label (e.g. a `Codicon`). */
  icon?: ReactNode
  label: ReactNode
}

/**
 * One collapsible group inside the composer status stack. Pure chrome — header
 * (caret + label) + body — styled to match the queue exactly so every status
 * (queue, subagents, background) reads as one piece. The stack supplies the
 * outer card and the dividers between groups; this owns only its own collapse.
 */
export function StatusSection({
  accessory,
  children,
  collapsedIndicator,
  defaultCollapsed = true,
  icon,
  label
}: StatusSectionProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed)

  return (
    <div>
      <div className="flex items-center gap-1 pr-1">
        <button
          className="flex min-w-0 flex-1 items-center gap-1.5 px-2 py-1 text-left text-xs font-normal text-muted-foreground/92 transition-colors hover:text-foreground/90"
          onClick={() => setCollapsed(open => !open)}
          type="button"
        >
          <DisclosureCaret className="shrink-0" open={!collapsed} size="1em" />
          {icon && <span className="flex shrink-0 items-center">{icon}</span>}
          <span className="min-w-0 truncate">{label}</span>
          {collapsed && collapsedIndicator && (
            <span className="flex shrink-0 items-center">{collapsedIndicator}</span>
          )}
        </button>
        {accessory && <div className="flex shrink-0 items-center gap-1">{accessory}</div>}
      </div>
      {!collapsed && <div className="px-1 pb-0.5">{children}</div>}
    </div>
  )
}
