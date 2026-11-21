import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect } from 'react'
import { useNavigate } from 'react-router'

import type { SidebarProjectTree } from '@/app/chat/sidebar/projects/workspace-groups'
import { SETTINGS_ROUTE } from '@/app/routes'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenuItem,
  DropdownMenuLabel,
  dropdownMenuRow,
  dropdownMenuSectionLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n'
import { displayPath } from '@/lib/display-path'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import {
  $defaultProjectId,
  $projectsRpcAvailable,
  $projectTree,
  goToProject,
  moveSessionToProject,
  openFolderAsProject,
  projectWorkspaceCwds,
  refreshProjectTree,
  setDefaultProject
} from '@/store/projects'

import type { StatusbarMenuItem } from './statusbar-controls'

export interface WorkspaceSwitcherMenuProps {
  /** A turn is running on the focused chat: moves are refused. */
  busy: boolean
  /** The existing "Change working directory" statusbar item, re-rendered as a row. */
  changeWorkspaceItem: StatusbarMenuItem
  /** The existing "Copy path" statusbar item. */
  copyPathItem: StatusbarMenuItem
  currentCwd: string
  currentProjectId: null | string
  /** Dismiss the open statusbar popover (render-fn `menuContent`'s `close`). */
  onClose: () => void
  /**
   * The active gateway profile a move writes through. Threaded in from the
   * statusbar's memoized menu content so a profile switch rebuilds the menu
   * instead of the move firing on the profile the menu was built under.
   */
  profile: null | string
  /** Projects-menu copy in the current locale, threaded through for the same reason. */
  projectsCopy: Translations['sidebar']['projects']
  /** Existing reveal rows (file-manager reveal is gated upstream by locality). */
  revealItems: readonly StatusbarMenuItem[]
  /** The FOCUSED stored session — the chat a "switch" would move. */
  sessionId: null | string
  /** The focused session's workspace lives on another machine/gateway. */
  sessionRemote: boolean
}

const MANAGE_PROJECTS_ROUTE = `${SETTINGS_ROUTE}?tab=config:workspace&page=projects`

// Rows the switcher can act on: projects owned by another gateway (surfaced in
// the all-profiles tree with a connectionId) would send writes to the wrong
// backend, and archived ones are out of the working set.
const switchableProjects = (tree: readonly SidebarProjectTree[]): SidebarProjectTree[] =>
  tree.filter(node => !node.connectionId && !node.archived)

const canBeDefault = (node: SidebarProjectTree): boolean => !node.isAuto && !node.isNoProject

/** Consistent left-aligned leading icon slot so labelled rows align. */
const iconSlot = (icon: ReactNode) => (
  <span className="flex w-4 shrink-0 items-center justify-start" data-icon-slot="true">
    {icon}
  </span>
)

/**
 * The workspace-cwd statusbar item's popover: a native project/workspace
 * switcher. Click one opens the list; a project row opens a submenu that
 * separates "Switch current chat" (move the focused chat, idle only), "Open
 * project" (scope only), "New chat" (fresh session, explicit folder for
 * multi-root) and "Default for new chats" (future-chats pointer only). The
 * footer keeps the folder/manage entries and the pre-existing cwd actions.
 */
export function WorkspaceSwitcherMenu({
  busy,
  changeWorkspaceItem,
  copyPathItem,
  currentCwd,
  currentProjectId,
  onClose,
  profile,
  projectsCopy,
  revealItems,
  sessionId,
  sessionRemote
}: WorkspaceSwitcherMenuProps) {
  const { t } = useI18n()
  const p = projectsCopy
  const navigate = useNavigate()
  const tree = useStore($projectTree)
  const defaultId = useStore($defaultProjectId)
  const rpcAvailable = useStore($projectsRpcAvailable)

  // Opening the switcher must show live truth regardless of sidebar grouping
  // state, the same way the session row's move-to-project submenu refreshes.
  useEffect(() => {
    void refreshProjectTree()
  }, [])

  const projects = switchableProjects(tree)
  const stale = rpcAvailable === false
  // Writes (move/default) need a stored chat on THIS gateway and a projects.*
  // backend. A busy chat is guarded per-action so the guidance stays visible.
  const showMove = Boolean(sessionId) && !sessionRemote

  const moveChat = (node: SidebarProjectTree, cwd?: string) => {
    if (!sessionId) {
      return
    }

    moveSessionToProject(sessionId, node.id, profile, cwd)
      .then(() => {
        onClose()
        notify({ durationMs: 2_000, kind: 'success', message: p.movedTo(node.label) })
      })
      .catch(err => notifyError(err, p.moveFailed))
  }

  const startNewChat = (node: SidebarProjectTree, cwd?: string) => {
    onClose()
    goToProject(node.id, { cwd: cwd || undefined, newSession: true })
  }

  // Only real saved projects (!isAuto && !isNoProject) own a durable row the
  // default pointer can reference; synthetic Home/auto nodes must never call
  // projects.set_default (the backend rejects them with "no such project").
  const toggleDefault = (node: SidebarProjectTree) => {
    if (!canBeDefault(node)) {
      return
    }

    setDefaultProject(defaultId === node.id ? null : node.id).catch(err => notifyError(err, p.setDefaultFailed))
  }

  const manageProjects = () => {
    onClose()
    navigate(MANAGE_PROJECTS_ROUTE)
  }

  return (
    <div className="flex flex-col py-1" role="presentation">
      {busy && showMove && <p className={cn('px-2.5 py-1', dropdownMenuSectionLabel)}>{p.moveBusy}</p>}
      {stale && <p className={cn('px-2.5 py-1', dropdownMenuSectionLabel)}>{p.staleBackend}</p>}

      {projects.length === 0 ? (
        <DropdownMenuLabel className={cn(dropdownMenuSectionLabel, 'normal-case')}>{p.noProjects}</DropdownMenuLabel>
      ) : (
        projects.map(node => {
          const cwds = projectWorkspaceCwds(node)
          const primaryCwd = cwds[0] ?? ''

          return (
            <div
              className="flex items-stretch"
              data-project-row={node.id}
              data-testid={`project-row-${node.id}`}
              key={node.id}
              role="presentation"
            >
              <DropdownMenuSub>
                <DropdownMenuSubTrigger
                  aria-current={node.id === currentProjectId ? 'true' : undefined}
                  className={cn(dropdownMenuRow, 'flex min-w-0 flex-1 items-center gap-2')}
                  title={displayPath(primaryCwd)}
                >
                  {iconSlot(
                    <Codicon
                      name={node.icon || (node.isNoProject ? 'home' : node.isAuto ? 'repo' : 'folder-library')}
                      size="0.75rem"
                      style={node.color ? { color: node.color } : undefined}
                    />
                  )}
                  <span className="flex min-w-0 flex-1 flex-col">
                    <span className="flex items-center gap-1 truncate text-xs text-foreground">{node.label}</span>
                    {primaryCwd && (
                      <span className="truncate text-[0.625rem] text-(--ui-text-tertiary)">
                        {displayPath(primaryCwd)}
                      </span>
                    )}
                  </span>
                  {node.id === currentProjectId && <Codicon aria-hidden="true" name="check" size="0.75rem" />}
                </DropdownMenuSubTrigger>
                <DropdownMenuSubContent className="w-60 p-1" sideOffset={4}>
                  <WorkspaceProjectActions
                    busy={busy}
                    currentProjectId={currentProjectId}
                    cwds={cwds}
                    defaultId={defaultId}
                    node={node}
                    onMove={moveChat}
                    onNewChat={startNewChat}
                    onToggleDefault={toggleDefault}
                    projectsCopy={p}
                    showMove={showMove}
                    stale={stale}
                  />
                </DropdownMenuSubContent>
              </DropdownMenuSub>
            </div>
          )
        })
      )}

      <DropdownMenuSeparator />
      {[changeWorkspaceItem, copyPathItem, ...revealItems]
        .filter(item => !item.hidden)
        .map(item => (
          <DropdownMenuItem
            className={dropdownMenuRow}
            disabled={item.disabled}
            key={item.id}
            onSelect={() => {
              item.onSelect?.()

              if (item.to) {
                navigate(item.to)
              }
            }}
            title={item.title}
          >
            {iconSlot(item.icon ?? null)}
            <span className="truncate">{item.label}</span>
          </DropdownMenuItem>
        ))}

      <DropdownMenuSeparator />
      <DropdownMenuItem
        className={dropdownMenuRow}
        onSelect={() => {
          onClose()
          void openFolderAsProject()
        }}
      >
        {iconSlot(<Codicon aria-hidden="true" name="folder" size="0.75rem" />)}
        <span className="truncate">{t.commandCenter.openFolder}</span>
      </DropdownMenuItem>
      <DropdownMenuItem className={dropdownMenuRow} onSelect={manageProjects}>
        {iconSlot(<Codicon aria-hidden="true" name="gear" size="0.75rem" />)}
        <span className="truncate">{p.manageProjects}</span>
      </DropdownMenuItem>
    </div>
  )
}

interface WorkspaceProjectActionsProps {
  busy: boolean
  currentProjectId: null | string
  cwds: string[]
  defaultId: null | string
  node: SidebarProjectTree
  onMove: (node: SidebarProjectTree, cwd?: string) => void
  onNewChat: (node: SidebarProjectTree, cwd?: string) => void
  onToggleDefault: (node: SidebarProjectTree) => void
  projectsCopy: Translations['sidebar']['projects']
  showMove: boolean
  stale: boolean
}

function WorkspaceProjectActions({
  busy,
  currentProjectId,
  cwds,
  defaultId,
  node,
  onMove,
  onNewChat,
  onToggleDefault,
  projectsCopy: p,
  showMove,
  stale
}: WorkspaceProjectActionsProps) {
  const multiRoot = cwds.length > 1
  const primaryCwd = cwds[0] ?? ''
  const moveDisabled = busy || stale
  const moveReason = stale ? p.staleBackend : busy ? p.moveBusy : null
  const isDefault = defaultId === node.id

  return (
    <>
      {/* Moving into the project the chat already occupies is a no-op on the
          only root — keep scope/new-chat. */}
      {showMove && cwds.length > 0 && !multiRoot && node.id !== currentProjectId && (
        <DropdownMenuItem
          className={dropdownMenuRow}
          disabled={moveDisabled}
          onSelect={() => onMove(node)}
          title={moveReason ?? displayPath(primaryCwd)}
        >
          {iconSlot(<Codicon aria-hidden="true" name="arrow-right" size="0.75rem" />)}
          <span className="truncate">{p.switchCurrentChat}</span>
        </DropdownMenuItem>
      )}
      {showMove && multiRoot && (
        <DropdownMenuSub>
          <DropdownMenuSubTrigger
            className={cn(dropdownMenuRow, 'flex w-full items-center gap-2')}
            disabled={moveDisabled}
          >
            {iconSlot(<Codicon aria-hidden="true" name="arrow-right" size="0.75rem" />)}
            <span className="truncate">{p.switchCurrentChat}</span>
          </DropdownMenuSubTrigger>
          <DropdownMenuSubContent className="w-60 p-1" sideOffset={4}>
            {cwds.map(cwd => (
              <DropdownMenuItem
                className={dropdownMenuRow}
                disabled={moveDisabled}
                key={cwd}
                onSelect={() => onMove(node, cwd)}
                title={moveReason ?? displayPath(cwd)}
              >
                {iconSlot(null)}
                <span className="truncate">{displayPath(cwd)}</span>
              </DropdownMenuItem>
            ))}
          </DropdownMenuSubContent>
        </DropdownMenuSub>
      )}

      {multiRoot ? (
        <DropdownMenuSub>
          <DropdownMenuSubTrigger className={cn(dropdownMenuRow, 'flex w-full items-center gap-2')}>
            {iconSlot(<Codicon aria-hidden="true" name="add" size="0.75rem" />)}
            <span className="truncate">{p.newChat}</span>
          </DropdownMenuSubTrigger>
          <DropdownMenuSubContent className="w-60 p-1" sideOffset={4}>
            {cwds.map(cwd => (
              <DropdownMenuItem
                className={dropdownMenuRow}
                key={`new:${cwd}`}
                onSelect={() => onNewChat(node, cwd)}
              >
                {iconSlot(null)}
                <span className="truncate">{displayPath(cwd)}</span>
              </DropdownMenuItem>
            ))}
          </DropdownMenuSubContent>
        </DropdownMenuSub>
      ) : (
        <DropdownMenuItem className={dropdownMenuRow} onSelect={() => onNewChat(node, primaryCwd)}>
          {iconSlot(<Codicon aria-hidden="true" name="add" size="0.75rem" />)}
          <span className="truncate">{p.newChat}</span>
        </DropdownMenuItem>
      )}

      <DropdownMenuItem className={dropdownMenuRow} onSelect={() => goToProject(node.id)}>
        {iconSlot(<Codicon aria-hidden="true" name="folder-opened" size="0.75rem" />)}
        <span className="truncate">{p.openProject}</span>
      </DropdownMenuItem>

      {canBeDefault(node) && (
        <>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            aria-pressed={isDefault}
            className={cn(
              dropdownMenuRow,
              isDefault ? 'text-warning' : 'text-(--ui-text-secondary)'
            )}
            disabled={stale}
            onSelect={() => onToggleDefault(node)}
            title={stale ? p.staleBackend : p.defaultForNewChats}
          >
            {iconSlot(
              <Codicon aria-hidden="true" name={isDefault ? 'star-full' : 'star-empty'} size="0.75rem" />
            )}
            <span className="truncate">{p.defaultForNewChats}</span>
          </DropdownMenuItem>
        </>
      )}
    </>
  )
}
