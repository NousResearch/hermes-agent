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

/** Figma 147:7470's section label: an uppercase caption on the row inset, then at the end an
 *  optional count badge, the disclosure chevron and a trailing action. */
export function PortalSectionHeader({ label, open, onToggle, count, action }: PortalSectionHeaderProps) {
  const trailing = (
    <>
      {count !== undefined && count > 0 && (
        <span aria-hidden className="portal-section-count">
          {count}
        </span>
      )}
      {onToggle && <ChevronDown aria-hidden className="portal-section-chevron" />}
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
