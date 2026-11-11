import type { ReactNode } from 'react'

import { RowButton } from '@/components/ui/row-button'
import { ChevronDown } from '@/lib/icons'

interface PortalSectionHeaderProps {
  label: string
  /** Present only for collapsible sections; the header then becomes the disclosure button. */
  open?: boolean
  onToggle?: () => void
  action?: ReactNode
}

/** Figma's 28px section label: quiet title, disclosure chevron, trailing 16px action. */
export function PortalSectionHeader({ label, open, onToggle, action }: PortalSectionHeaderProps) {
  return (
    <div className="portal-section-header" data-slot="sidebar-section-label">
      {onToggle ? (
        <RowButton aria-expanded={open} className="portal-section-title" onClick={onToggle}>
          <span className="truncate">{label}</span>
          <ChevronDown aria-hidden className="portal-section-chevron" />
        </RowButton>
      ) : (
        <span className="portal-section-title">
          <span className="truncate">{label}</span>
        </span>
      )}
      {action}
    </div>
  )
}
