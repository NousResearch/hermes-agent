import type * as React from 'react'

import { DisclosureCaret } from '@/components/ui/disclosure-caret'

import { SidebarPanelLabel } from '../../shell/sidebar-label'

/** Count badge shared by every sidebar section header and group header, so all
 *  the little numbers in the rail read as one family. */
export function SidebarCount({ children }: { children: React.ReactNode }) {
  return <span className="text-[0.6875rem] font-medium text-(--ui-text-quaternary)">{children}</span>
}

export interface SidebarSectionHeaderProps {
  label: string
  open: boolean
  onToggle: () => void
  action?: React.ReactNode
  meta?: React.ReactNode
  icon?: React.ReactNode
}

/** The one section header for the chat sidebar (Pinned / Sessions / platforms /
 *  Cron / Live elsewhere / Archived / search results).
 *
 *  Geometry contract, shared with the session/cron/remote rows below it:
 *  `pl-2` + a `w-3.5` leading slot + `gap-1.5` — so the dither dot (or the
 *  section's identity icon, via `icon`) centers on the same column as the row
 *  status dots, and the label's left edge sits exactly on the rows' title
 *  edge. `min-h-[1.875rem]` keeps every header the same height whether or not
 *  it carries trailing actions. */
export function SidebarSectionHeader({ label, open, onToggle, action, meta, icon }: SidebarSectionHeaderProps) {
  return (
    <div className="group/section flex min-h-[1.875rem] shrink-0 items-center justify-between pb-1 pt-1.5">
      <button
        className="group/section-label flex w-fit min-w-0 items-center gap-1.5 bg-transparent text-left leading-none"
        onClick={onToggle}
        type="button"
      >
        <SidebarPanelLabel className="gap-1.5" glyph={icon} slotClassName="w-3.5">
          {label}
        </SidebarPanelLabel>
        {meta && <SidebarCount>{meta}</SidebarCount>}
        <DisclosureCaret
          className="text-(--ui-text-tertiary) opacity-0 transition group-hover/section-label:opacity-100"
          open={open}
        />
      </button>
      {action}
    </div>
  )
}
