import type {
  CSSProperties,
  ComponentProps,
  ComponentType,
  HTMLAttributes,
  PointerEvent,
  ReactNode,
  Ref,
  RefObject
} from 'react'

import type { NewSessionPlacement, NewSessionSplitHandler } from '@/app/chat/new-session-drag'

import type { SidebarProjectTree, SidebarSessionGroup } from './workspace-groups'

export type ProjectOverviewPresentation = ComponentType<{
  ref?: Ref<HTMLDivElement>
  style?: CSSProperties
  model: {
    project: SidebarProjectTree
    isActive: boolean
    open: boolean
    toggleOpen(): void
    onEnter?(id: string): void
    onNewSession?(path: string | null): void
    onNewSessionSplit?: NewSessionSplitHandler
    newSessionPath: string | null
    rowRef: RefObject<HTMLDivElement | null>
    reorderable: boolean
    dragging: boolean
    dragHandleProps?: HTMLAttributes<HTMLElement>
    hasRows: boolean
    rows: ReactNode
  }
}>

export type WorkspaceGroupPresentation = ComponentType<{
  model: {
    group: SidebarSessionGroup
    isProfileGroup: boolean
    usage?: { tokens: number; cost_usd: number }
    open: boolean
    toggleOpen(): void
    empty: boolean
    rows: ReactNode
    canCreate: boolean
    handleNewSession(): Promise<void>
    handleNewSessionSplit(placement: NewSessionPlacement): Promise<void>
    onNewSessionSplit?: NewSessionSplitHandler
    onRemove?(): void
  }
}>

export type RepositoryPresentation = ComponentType<{
  model: {
    label: string
    title?: string
    open: boolean
    toggle(): void
    emphasis: boolean
    icon: ReactNode
    actions: ReactNode
  }
  dialogs: ReactNode
  children: ReactNode
}>

export interface WorkspaceChromePresentation {
  Add: ComponentType<{ label: string; onClick(): void; onPointerDown?(event: PointerEvent<HTMLButtonElement>): void }>
  More: ComponentType<{ label: string; onClick(): void; disabled?: boolean; compact?: boolean }>
  Start: ComponentType<{ label: string; onClick(): void }>
  Header: ComponentType<
    ComponentProps<'div'> & {
      model: {
        label: string
        title?: string
        open: boolean
        toggle(): void
        emphasis: boolean
        icon: ReactNode
        actions?: ReactNode
      }
    }
  >
}
