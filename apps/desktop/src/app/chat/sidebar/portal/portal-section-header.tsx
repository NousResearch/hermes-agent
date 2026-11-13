import type { ReactNode } from 'react'

import { RowButton } from '@/components/ui/row-button'
import { ChevronDown } from '@/lib/icons'

interface PortalSectionHeaderProps {
  label: string
  /** Present only for collapsible sections; the header then becomes the disclosure button. */
  open?: boolean
  onToggle?: () => void
  /** A count badge beside the disclosure (Agents, 147:6259). */
  count?: number
  action?: ReactNode
}

/** A section's caption (label 16292:15524): the label on the row inset with the disclosure chevron
 *  right after it, then at the end an optional count badge and a trailing action. */
export function PortalSectionHeader({ label, open, onToggle, count, action }: PortalSectionHeaderProps) {
  const trailing = (
    <>
      {onToggle && <ChevronDown aria-hidden className="portal-section-chevron" />}
      {count !== undefined && count > 0 && (
        <span aria-hidden className="portal-section-count">
          {count}
        </span>
      )}
    </>
  )

  return (
    <div className="portal-section-header" data-slot="sidebar-section-label">
      {onToggle ? (
        <RowButton aria-expanded={open} className="portal-section-title" onClick={onToggle}>
          <span className="portal-section-label">{label}</span>
          {trailing}
        </RowButton>
      ) : (
        <span className="portal-section-title">
          <span className="portal-section-label">{label}</span>
          {trailing}
        </span>
      )}
      {action}
    </div>
  )
}
