import { useStore } from '@nanostores/react'
import { useMemo, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { releaseTypingFocus } from '@/components/ui/keyboard-first'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { selectDesktopPaths } from '@/lib/desktop-fs'
import { displayPath, pathLeaf } from '@/lib/display-path'
import { Check, ChevronDown, FolderOpen, Pencil, Plus, Zap, ZapFilled } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { requestGatewayForActiveRoute } from '@/store/active-route-request'
import { $approvalModes, type ApprovalMode, setApprovalModeForProfile } from '@/store/approval-mode'
import { copyFilePath } from '@/store/file-actions'
import { revealFileInTree } from '@/store/layout'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $projectTree,
  openProjectAddFolder,
  openProjectCreate,
  openProjectRename,
  projectIdForCwd,
  projectNameForCwd,
  projectRootCwd,
  refreshProjectTree,
  setActiveProject
} from '@/store/projects'
import { $currentCwd, $gatewayState } from '@/store/session'

const PILL = cn(
  'h-(--composer-control-size) min-w-0 shrink gap-1 rounded-md px-2 text-xs font-normal',
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)

export interface ComposerWorkspaceControlsProps {
  busy: boolean
  cwd?: null | string
  disabled: boolean
  onChangeCwd?: (cwd: string, sessionId?: string) => Promise<void>
  profile?: null | string
  sessionId?: null | string
}

/** Project / working directory picker + Approval mode selector in the redesigned composer top row. */
export function ComposerWorkspaceControls({
  busy,
  cwd: cwdProp,
  disabled,
  onChangeCwd,
  profile,
  sessionId
}: ComposerWorkspaceControlsProps) {
  return (
    <div className="flex min-w-0 items-center gap-1" data-slot="composer-workspace-controls">
      <ComposerProjectPill
        busy={busy}
        cwd={cwdProp}
        disabled={disabled}
        onChangeCwd={onChangeCwd}
        sessionId={sessionId}
      />
      <ComposerApprovalPill disabled={disabled} profile={profile} />
    </div>
  )
}

function ComposerProjectPill({
  busy,
  cwd: cwdProp,
  disabled,
  onChangeCwd,
  sessionId
}: {
  busy: boolean
  cwd?: null | string
  disabled: boolean
  onChangeCwd?: (cwd: string, sessionId?: string) => Promise<void>
  sessionId?: null | string
}) {
  const { t } = useI18n()
  const view = useSessionView()
  const viewCwd = useStore(view.$cwd)
  const globalCwd = useStore($currentCwd)
  const projects = useStore($projectTree)
  const [open, setOpen] = useState(false)

  const effectiveCwd = (cwdProp || viewCwd || globalCwd || '').trim()

  if (!effectiveCwd) {
    return null
  }

  const projectName = projectNameForCwd(effectiveCwd)
  const activeProjectId = projectIdForCwd(effectiveCwd)
  const activeProject = activeProjectId ? projects.find(project => project.id === activeProjectId) : undefined
  const label = projectName || pathLeaf(effectiveCwd)
  const fullPath = displayPath(effectiveCwd)
  const pickableProjects = projects
    .map(project => ({ cwd: projectRootCwd(project), project }))
    .filter(entry => Boolean(entry.cwd))

  const setMenuOpen = (next: boolean) => {
    setOpen(next)

    if (next && projects.length === 0) {
      void refreshProjectTree()
    }

    if (!next) {
      releaseTypingFocus()
    }
  }

  const pickDirectory = async () => {
    if (!onChangeCwd) {
      return
    }

    const [dir] = await selectDesktopPaths({
      defaultPath: effectiveCwd,
      directories: true,
      multiple: false,
      title: t.rightSidebar.changeCwdTitle
    })

    if (dir) {
      await onChangeCwd(dir, sessionId ?? undefined)
    }
  }

  return (
    <DropdownMenu onOpenChange={setMenuOpen} open={open}>
      <Tip label={fullPath} side="top">
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={`${t.shell.statusbar.toggleWorkspace}: ${label}`}
            className={PILL}
            data-testid="composer-project-pill"
            disabled={disabled}
            type="button"
            variant="ghost"
          >
            <FolderOpen className="size-3 shrink-0 opacity-70" />
            <span className="truncate">{label}</span>
            <ChevronDown className="size-2.5 shrink-0 opacity-50" />
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="start" className="w-64 p-1" side="top" sideOffset={8}>
        <DropdownMenuLabel className="truncate font-mono text-[0.6875rem] font-normal text-(--ui-text-tertiary)">
          {fullPath}
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        {onChangeCwd && pickableProjects.length > 0 && (
          <>
            {pickableProjects.slice(0, 8).map(({ cwd: projectCwd, project }) => {
              const isCurrent = project.id === activeProjectId

              return (
                <DropdownMenuItem
                  disabled={busy}
                  key={project.id}
                  onSelect={() => {
                    if (!project.isAuto) {
                      void setActiveProject(project.id)
                    }

                    void onChangeCwd(projectCwd, sessionId ?? undefined)
                  }}
                >
                  <FolderOpen className="size-3.5 shrink-0 opacity-70" />
                  <span className="min-w-0 flex-1 truncate">{project.label}</span>
                  {isCurrent && <Check className="size-3.5 shrink-0 opacity-80" />}
                </DropdownMenuItem>
              )
            })}
            <DropdownMenuSeparator />
          </>
        )}
        {onChangeCwd && (
          <DropdownMenuItem disabled={busy} onSelect={() => void pickDirectory()}>
            <FolderOpen className="size-3.5 shrink-0 opacity-70" />
            <span>{t.rightSidebar.changeCwdTitle}</span>
          </DropdownMenuItem>
        )}
        <DropdownMenuItem onSelect={() => openProjectCreate()}>
          <Plus className="size-3.5 shrink-0 opacity-70" />
          <span>{t.sidebar.projects.newButton}</span>
        </DropdownMenuItem>
        {activeProject && !activeProject.isAuto && (
          <>
            <DropdownMenuItem
              onSelect={() => openProjectRename({ id: activeProject.id, name: activeProject.label })}
            >
              <Pencil className="size-3.5 shrink-0 opacity-70" />
              <span>{t.sidebar.projects.menuRename}</span>
            </DropdownMenuItem>
            <DropdownMenuItem
              onSelect={() => openProjectAddFolder({ id: activeProject.id, name: activeProject.label })}
            >
              <FolderOpen className="size-3.5 shrink-0 opacity-70" />
              <span>{t.sidebar.projects.menuAddFolder}</span>
            </DropdownMenuItem>
          </>
        )}
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={() => void copyFilePath(effectiveCwd)}>
          <span>{t.fileMenu.copyPath}</span>
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={() => revealFileInTree(effectiveCwd)}>
          <span>{t.fileMenu.revealInSidebar}</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function ComposerApprovalPill({ disabled, profile }: { disabled: boolean; profile?: null | string }) {
  const { t } = useI18n()
  const copy = t.shell.approvalMode
  const activeProfile = useStore($activeGatewayProfile)
  const gatewayState = useStore($gatewayState)
  const modes = useStore($approvalModes)
  const [open, setOpen] = useState(false)

  const targetProfile = (profile || activeProfile || 'default').trim() || 'default'
  const mode: ApprovalMode = modes[targetProfile] ?? 'smart'

  const labels = useMemo<Record<ApprovalMode, string>>(
    () => ({ manual: copy.manual, smart: copy.smart, off: copy.off }),
    [copy.manual, copy.off, copy.smart]
  )

  const descriptions = useMemo<Record<ApprovalMode, string>>(
    () => ({
      manual: copy.manualDescription,
      smart: copy.smartDescription,
      off: copy.offDescription
    }),
    [copy.manualDescription, copy.offDescription, copy.smartDescription]
  )

  const setMenuOpen = (next: boolean) => {
    setOpen(next)

    if (!next) {
      releaseTypingFocus()
    }
  }

  const label = labels[mode]
  const title = `${copy.title}: ${label}`

  return (
    <DropdownMenu onOpenChange={setMenuOpen} open={open}>
      <Tip label={title} side="top">
        <DropdownMenuTrigger asChild>
          <Button
            aria-label={title}
            className={cn(PILL, mode === 'off' && 'text-foreground')}
            data-testid="composer-approval-pill"
            disabled={disabled || gatewayState !== 'open'}
            type="button"
            variant="ghost"
          >
            {mode === 'off' ? (
              <ZapFilled className="size-3 shrink-0" />
            ) : (
              <Zap className="size-3 shrink-0 opacity-70" />
            )}
            <span>{label}</span>
            <ChevronDown className="size-2.5 shrink-0 opacity-50" />
          </Button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="start" className="w-72 p-1" side="top" sideOffset={8}>
        <DropdownMenuLabel>{copy.title}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuRadioGroup
          onValueChange={value => {
            void setApprovalModeForProfile(
              (method, params) => requestGatewayForActiveRoute(targetProfile, method, params),
              targetProfile,
              value as ApprovalMode
            ).catch(() => undefined)
          }}
          value={mode}
        >
          {(['manual', 'smart', 'off'] as const).map(value => (
            <DropdownMenuRadioItem className="items-start gap-2" key={value} value={value}>
              <div className="min-w-0">
                <div className="font-medium text-foreground">{labels[value]}</div>
                <div className="text-[0.6875rem] leading-snug text-(--ui-text-tertiary)">{descriptions[value]}</div>
              </div>
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
