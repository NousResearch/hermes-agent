import type { ComponentProps, ComponentType, ReactNode } from 'react'

import type { NewSessionSplitHandler } from '@/app/chat/new-session-drag'
import type { SessionInfo } from '@/hermes'

import type { SidebarSessionGroup } from './projects/workspace-groups'
import type { ReorderableList, useSortableBindings } from './reorderable-list'

export interface ProfileGroupPresentation {
  flatten: boolean
  hiddenProfiles: readonly string[]
  filterHeaderContribution?(id: string): boolean
  Groups: ComponentType<{
    model: {
      ids: string[]
      ordered: SidebarSessionGroup[]
      gatewayProfiles: Map<string, SidebarSessionGroup[]>
      sensors?: ComponentProps<typeof ReorderableList>['sensors']
    }
    renderGroup(
      group: SidebarSessionGroup,
      index: number,
      move: (direction: 1 | -1) => void,
      children: ReactNode
    ): ReactNode
    renderNested(groups: SidebarSessionGroup[]): ReactNode
  }>
  Group: ComponentType<{
    model: {
      group: SidebarSessionGroup
      label: string
      open: boolean
      embedded: boolean
      first: boolean
      last: boolean
      alias?: string
      sortable: ReturnType<typeof useSortableBindings>
      usage?: { tokens: number; cost_usd: number }
      renaming: boolean
      setRenaming(next: boolean): void
      draft: string
      setDraft(next: string): void
      onMove(direction: 1 | -1): void
      startSession(): void
      onNewSessionSplit?: NewSessionSplitHandler
      route?: { connectionId: string; profile: string }
      sessions: SessionInfo[]
      renderRows(sessions: SessionInfo[]): ReactNode
      contribution: ReactNode
      children: ReactNode
    }
  }>
}
