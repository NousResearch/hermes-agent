import type { ComponentProps, ComponentType, ReactNode } from 'react'

import type { SessionInfo } from '@/hermes'
import type { SidebarListRow } from '@/lib/session-date-groups'
import type { SessionDotState } from '@/store/session-dot-state'

import type { SidebarDateDivider } from './chrome'
import type { SidebarProjectTree, SidebarSessionGroup } from './projects'
import type { ReorderableList } from './reorderable-list'
import type { SidebarSessionsSectionProps } from './sessions-section'

export interface SidebarSessionSectionView {
  sectionId?: string
  label: string
  open: boolean
  collapsible: boolean
  toggle(): void
  icon?: ReactNode
  meta?: ReactNode
  actions?: ReactNode
  className?: string
  contentClassName?: string
  pinned: boolean
}

interface ProfileProjectsProps {
  sessions: SessionInfo[]
  projects: SidebarProjectTree[]
  renderRows(sessions: SessionInfo[]): ReactNode
  activeProjectId?: string | null
  onEnterProject?: SidebarSessionsSectionProps['onEnterProject']
  onNewSession?: SidebarSessionsSectionProps['onNewSessionInWorkspace']
  onNewSessionSplit?: SidebarSessionsSectionProps['onNewSessionSplit']
}

/** Optional host presentation, sharing the renderer's list and ordering controller. */
export interface SidebarSessionsSectionPresentation {
  Section: ComponentType<{ model: SidebarSessionSectionView; children: ReactNode; footer?: ReactNode }>
  SessionReorderList?: ComponentType<ComponentProps<typeof ReorderableList>>
  Divider?: ComponentType<ComponentProps<typeof SidebarDateDivider>>
  reorderGroupedSessions?: boolean
  profileRows?(
    sessions: SessionInfo[],
    grouping: NonNullable<SidebarSessionsSectionProps['grouping']>,
    states: Record<string, SessionDotState>,
    labels: { done: string; working: string }
  ): SidebarListRow[]
  profileProjects?: {
    Render: ComponentType<ProfileProjectsProps>
    ids(groups: SidebarSessionGroup[], projects: SidebarProjectTree[]): string[]
  }
}
