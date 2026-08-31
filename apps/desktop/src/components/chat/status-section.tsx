import { type ReactNode, type Ref, useId, useState } from 'react'

import { DisclosureCaret } from '@/components/ui/disclosure-caret'

interface StatusSectionProps {
  /** Optional right-aligned actions (text links / micro buttons). Pass
   *  `Button` with `size="micro"` + `variant="text"` or `"link"`. */
  accessory?: ReactNode
  children: ReactNode
  /** Optional inline status shown only while the group is collapsed. */
  collapsedIndicator?: ReactNode
  collapsed?: boolean
  defaultCollapsed?: boolean
  /** Optional glyph between the caret and the label (e.g. a `Codicon`). */
  icon?: ReactNode
  label: ReactNode
  onCollapsedChange?: (collapsed: boolean) => void
  triggerRef?: Ref<HTMLButtonElement>
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
  collapsed,
  collapsedIndicator,
  defaultCollapsed = true,
  icon,
  label,
  onCollapsedChange,
  triggerRef
}: StatusSectionProps) {
  const [localCollapsed, setLocalCollapsed] = useState(defaultCollapsed)
  const effectiveCollapsed = collapsed ?? localCollapsed
  const bodyId = useId()

  const toggle = () => {
    const next = !effectiveCollapsed
    onCollapsedChange?.(next)

    if (collapsed === undefined) {
      setLocalCollapsed(next)
    }
  }

  return (
    <div>
      <div className="flex items-center gap-1 pr-1">
        <button
          aria-controls={effectiveCollapsed ? undefined : bodyId}
          aria-expanded={!effectiveCollapsed}
          className="flex min-w-0 flex-1 items-center gap-1.5 px-2 py-1 text-left text-xs font-normal text-muted-foreground/92 transition-colors hover:text-foreground/90"
          onClick={toggle}
          ref={triggerRef}
          type="button"
        >
          <DisclosureCaret className="shrink-0" open={!effectiveCollapsed} size="1em" />
          {icon && <span className="flex shrink-0 items-center">{icon}</span>}
          <span className="min-w-0 truncate">{label}</span>
          {effectiveCollapsed && collapsedIndicator && (
            <span className="flex shrink-0 items-center">{collapsedIndicator}</span>
          )}
        </button>
        {accessory && <div className="flex shrink-0 items-center gap-1">{accessory}</div>}
      </div>
      {!effectiveCollapsed && (
        <div className="px-1 pb-0.5" id={bodyId}>
          {children}
        </div>
      )}
    </div>
  )
}
