import type { ComponentType, ReactNode } from 'react'

export interface CronSidebarRowView {
  label: string
  meta: string
  stateClass: string
  running: boolean
  busy: boolean
  expanded: boolean
  toggleLabel: string
  triggerLabel: string
  manageLabel: string
  toggle(): void
  trigger(): void
  manage(): void
  wrap(header: ReactNode): ReactNode
}

export interface CronSidebarRunView {
  id: string
  label: string
  selected: boolean
  open?: () => void
}

/** The renderer owns scheduling, paging, and run selection. A host may supply their views. */
export interface CronSidebarPresentation {
  Section: ComponentType<{ label: string; open: boolean; toggle(): void; children: ReactNode }>
  Row: ComponentType<{ model: CronSidebarRowView; children: ReactNode }>
  Runs: ComponentType<{ rows: CronSidebarRunView[] | null; loadingLabel: string; emptyLabel: string }>
}
