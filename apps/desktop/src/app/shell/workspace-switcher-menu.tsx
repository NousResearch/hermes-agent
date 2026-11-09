import { useStore } from '@nanostores/react'
import { useEffect } from 'react'
import { useNavigate } from 'react-router'

import { projectIcon } from '@/app/chat/sidebar/projects/overview-row'
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

  const toggleDefault = (node: SidebarProjectTree) => {
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
          const multiRoot = cwds.length > 1
          const primaryCwd = cwds[0] ?? ''

          return (
            <DropdownMenuSub key={node.id}>
              <DropdownMenuSubTrigger
                aria-current={node.id === currentProjectId ? 'true' : undefined}
                className={cn(dropdownMenuRow, 'flex w-full items-center gap-2')}
                title={displayPath(primaryCwd)}
              >
                {projectIcon(node)}
                <span className="flex min-w-0 flex-1 flex-col">
                  <span className="flex items-center gap-1 truncate text-xs text-foreground">
                    {node.label}
                    {defaultId === node.id && (
                      <span title={p.defaultForNewChats}>
                        <Codicon aria-hidden="true" name="star-filled" size="0.625rem" />
                      </span>
                    )}
                  </span>
                  {primaryCwd && (
                    <span className="truncate text-[0.625rem] text-(--ui-text-tertiary)">
                      {displayPath(primaryCwd)}
                    </span>
                  )}
                </span>
                {node.id === currentProjectId && <Codicon aria-hidden="true" name="check" size="0.75rem" />}
              </DropdownMenuSubTrigger>
              <DropdownMenuSubContent className="w-60 p-1" sideOffset={4}>
                {/* Moving into the project the chat already occupies is a
                    no-op on the only root — keep scope/new-chat/default. */}
                {showMove && cwds.length > 0 && !multiRoot && node.id !== currentProjectId && (
                  <DropdownMenuItem
                    className={dropdownMenuRow}
                    disabled={busy || stale}
                    onSelect={() => moveChat(node)}
                    title={stale ? p.staleBackend : busy ? p.moveBusy : displayPath(primaryCwd)}
                  >
                    <span className="truncate">{p.switchCurrentChat}</span>
                  </DropdownMenuItem>
                )}
                {showMove && multiRoot && (
                  <>
                    <DropdownMenuLabel className={dropdownMenuSectionLabel}>{p.chooseFolder}</DropdownMenuLabel>
                    {cwds.map(cwd => (
                      <DropdownMenuItem
                        className={dropdownMenuRow}
                        disabled={busy || stale}
                        key={cwd}
                        onSelect={() => moveChat(node, cwd)}
                        title={stale ? p.staleBackend : busy ? p.moveBusy : displayPath(cwd)}
                      >
                        <span className="truncate">{displayPath(cwd)}</span>
                      </DropdownMenuItem>
                    ))}
                  </>
                )}

                {multiRoot ? (
                  <>
                    <DropdownMenuLabel className={dropdownMenuSectionLabel}>{p.newChat}</DropdownMenuLabel>
                    {cwds.map(cwd => (
                      <DropdownMenuItem
                        className={dropdownMenuRow}
                        key={`new:${cwd}`}
                        onSelect={() => {
                          onClose()
                          goToProject(node.id, { cwd, newSession: true })
                        }}
                      >
                        <span className="truncate">{displayPath(cwd)}</span>
                      </DropdownMenuItem>
                    ))}
                  </>
                ) : (
                  <DropdownMenuItem
                    className={dropdownMenuRow}
                    onSelect={() => {
                      onClose()
                      goToProject(node.id, { cwd: primaryCwd || undefined, newSession: true })
                    }}
                  >
                    <span className="truncate">{p.newChat}</span>
                  </DropdownMenuItem>
                )}

                <DropdownMenuItem className={dropdownMenuRow} onSelect={() => goToProject(node.id)}>
                  <span className="truncate">{p.openProject}</span>
                </DropdownMenuItem>

                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className={dropdownMenuRow}
                  disabled={stale}
                  onSelect={() => toggleDefault(node)}
                  title={stale ? p.staleBackend : undefined}
                >
                  <Codicon aria-hidden="true" name={defaultId === node.id ? 'star-filled' : 'star'} size="0.75rem" />
                  <span className="truncate">{p.defaultForNewChats}</span>
                </DropdownMenuItem>
              </DropdownMenuSubContent>
            </DropdownMenuSub>
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
            {item.icon}
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
        <span className="truncate">{t.commandCenter.openFolder}</span>
      </DropdownMenuItem>
      <DropdownMenuItem className={dropdownMenuRow} onSelect={manageProjects}>
        <span className="truncate">{p.manageProjects}</span>
      </DropdownMenuItem>
    </div>
  )
}
