import type { useSensors } from '@dnd-kit/core'
import { createContext, type ReactNode, type RefObject } from 'react'

import type { DesktopConnectionsRegistry, HermesGitWorktree } from '@/global'
import type { SessionInfo } from '@/hermes'
import type { Translations } from '@/i18n'
import type { $cronJobs } from '@/store/cron'
import type { SidebarGrouping } from '@/store/layout'
import type { SessionResumeRequest } from '@/store/session'

import type { ChatSidebarProps, MessagingSection } from './index'
import type { AppView } from '../../routes'
import type { SidebarNavItem } from '../../types'
import type { SidebarProjectTree, SidebarSessionGroup } from './projects'

/** The sidebar state and commands a host needs to replace its presentation. */
export interface ChatSidebarController extends Pick<
  ChatSidebarProps,
  | 'onNavigate'
  | 'onResumeSession'
  | 'onDeleteSession'
  | 'onArchiveSession'
  | 'onBranchSession'
  | 'onNewSessionInWorkspace'
  | 'onNewSessionSplit'
  | 'onRetrySessions'
  | 'onManageCronJob'
  | 'onTriggerCronJob'
> {
  labels: Translations['sidebar']
  navItems: SidebarNavItem[]
  pathname: string
  currentView: AppView
  panesFlipped: boolean
  showSessionSections: boolean
  searchQuery: string
  setSearchQuery(value: string): void
  searchInputRef: RefObject<HTMLInputElement | null>
  trimmedQuery: string
  searchPending: boolean
  searchResults: SessionInfo[]
  activeSidebarSessionId: string | null
  activeSidebarSessionOwner?: SessionResumeRequest['ownerRoute']
  showAllProfiles: boolean
  cardRows: boolean
  dndSensors: ReturnType<typeof useSensors>
  pinnedSessions: SessionInfo[]
  pinsOpen: boolean
  agentsOpen: boolean
  cronOpen: boolean
  inProject: boolean
  enteredProjectId?: string
  projectLoadFailed: boolean
  projectLoading: boolean
  retryProject(): void
  showSessionSkeletons: boolean
  sessionsLoadError: boolean
  filtersActive: boolean
  filtersNarrow: boolean
  sessionsLoading: boolean
  recentsLoadMorePending: boolean
  hasMoreSessions: boolean
  rankedGlobally: boolean
  combinedProfiles: boolean
  grouping: SidebarGrouping
  displayAgentGroups?: SidebarSessionGroup[]
  worktreeGroupingActive: boolean
  reposScanning: boolean
  projectsSkeletonVisible: boolean
  enteredProject?: SidebarProjectTree
  activeProjectId: string | null
  enteredProjectOverlaySessions: SessionInfo[]
  agentOrderManual: boolean
  agentOrderIds: string[]
  sortOrderIds: string[]
  projectOverview?: SidebarProjectTree[]
  overviewHidden: { isHidden(session: SessionInfo): boolean; counts: Record<string, number> }
  overviewPreviews: Record<string, SessionInfo[]>
  scopedRepoWorktrees: Record<string, HermesGitWorktree[]>
  enteredProjectContent?: SidebarProjectTree
  projectTreeLoading: boolean
  removedSessionIds: Set<string>
  displayAgentSessions: SessionInfo[]
  agentSessions: SessionInfo[]
  recentsVirtualizes: boolean
  sessionsMode: string
  sessionsLabel: string
  messagingGroups: MessagingSection[]
  ownerGrouped: boolean
  connectionsRegistry: DesktopConnectionsRegistry | null
  profileColors: Record<string, string>
  messagingLoadMorePending: Record<string, boolean>
  messagingOpenIds: string[]
  cronJobs: ReturnType<typeof $cronJobs.get>
  profileRailVisible: boolean
  unreadCount: number
  toggleUnread(id: string): void
  reorderPinned(ids: string[]): void
  reorderSessions(ids: string[]): void
  reorderProjects(ids: string[]): void
  onLoadMoreRecents(): Promise<void>
  onEnterProject(id: string): void
  loadMoreForMessaging(platform: string): void
}

export interface ChatSidebarPresentation {
  readonly groupAcrossProfiles?: boolean
  projectProfileGroups?(groups: SidebarSessionGroup[] | undefined): SidebarSessionGroup[] | undefined
  render(controller: ChatSidebarController): ReactNode
}

/** Internal renderer composition seam. The default desktop view uses the same controller. */
export const ChatSidebarPresentationContext = createContext<ChatSidebarPresentation | null>(null)
