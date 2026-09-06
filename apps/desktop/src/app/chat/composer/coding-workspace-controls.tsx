import { useHermesConfigRecord } from '@/app/hooks/use-config-record'
import { getNested } from '@/app/settings/helpers'
import { StatusRow } from '@/components/chat/status-row'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DropdownMenu, DropdownMenuContent, DropdownMenuLabel, DropdownMenuRadioGroup, DropdownMenuRadioItem, DropdownMenuSub, DropdownMenuSubContent, DropdownMenuSubTrigger, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { ErrorIcon } from '@/components/ui/error-state'
import { Loader } from '@/components/ui/loader'
import { useI18n } from '@/i18n'
import { pathLeaf } from '@/lib/display-path'
import { type CodingWorkspaceDraft, type CodingWorkspaceIntent, type CodingWorkspaceOwner } from '@/store/coding-workspaces'
import type { ProjectInfo } from '@/types/hermes'

import { CodingProjectPicker } from './coding-project-picker'
import { selectCodingWorkspaceIntent } from './coding-workspace-selection'
import { workspaceRowClassName } from './workspace-row'

interface CodingWorkspaceControlsProps {
  owner: CodingWorkspaceOwner
  draft?: CodingWorkspaceDraft
  onSelectFolder: (path?: string) => void
}

export function CodingWorkspaceControls({ owner, draft, onSelectFolder }: CodingWorkspaceControlsProps) {
  const { t } = useI18n()
  const c = t.codingWorkspace
  const { data: config } = useHermesConfigRecord({ connectionId: owner.connectionId, profile: owner.profile })
  const defaultCheckout = getNested(config ?? {}, 'desktop.coding.default_checkout') === 'current' ? 'current' : 'worktree'
  const intent = draft?.intent
  const inspection = draft?.inspection
  const branches = inspection?.branches ?? []
  const mainCheckout = inspection?.worktrees.find(tree => tree.isMain)
  const sharedPath = intent?.mode === 'current' ? mainCheckout?.path : intent?.mode === 'existing' ? intent.existingPath : null
  const activeChats = inspection?.worktrees.find(tree => tree.path === sharedPath)?.activeSessionCount ?? 0
  const locked = draft?.status === 'preparing' || Boolean(draft?.prepared || draft?.sessionId)
  const modeLabels = { worktree: c.newWorktree, existing: c.existingWorktree, current: c.currentCheckout, folder: c.projectFolder }
  const mode = inspection?.repoRoot ? intent?.mode ?? 'worktree' : 'folder'
  const base = intent?.base ?? inspection?.branch ?? 'HEAD'
  const selectedCheckout = inspection?.worktrees.find(tree => tree.path === intent?.existingPath)

  const checkoutLabel = mode === 'existing' && intent?.existingPath
    ? selectedCheckout?.branch || pathLeaf(intent.existingPath) : null

  const branchDetail = draft?.prepared?.branch ?? (base !== (inspection?.branch ?? 'HEAD') ? base : null)

  const modeSummary = mode === 'worktree' && branchDetail
    ? `${modeLabels[mode]} · ${branchDetail}` : modeLabels[mode]

  const workInLabel = checkoutLabel ? `${modeLabels[mode]} · ${checkoutLabel}` : modeSummary

  const update = async (next: CodingWorkspaceIntent | null) => {
    try { await selectCodingWorkspaceIntent(owner, next) } catch { /* The owner-bound draft paints the error. */ }
  }

  const selectProject = (project: ProjectInfo | null) => void update(project?.primary_path
    ? { projectId: project.id, path: project.primary_path, mode: defaultCheckout } : null)

  // `rounded-t-[inherit]` so the header strip inside inherits the composer's
  // top radius through this wrapper; the preparation error, when present,
  // hangs under the strip in the same inset and closes with its own hairline.
  return <div className="grid min-w-0 rounded-t-[inherit] text-xs text-(--ui-text-secondary)" data-slot="coding-workspace-controls">
    <StatusRow className={workspaceRowClassName} leading={
      draft?.status === 'inspecting' || draft?.status === 'preparing'
        ? <Loader className="size-3.5 text-(--ui-text-tertiary)" label={draft.status === 'preparing' ? c.preparing : c.project} />
        : <Codicon className="text-(--ui-text-tertiary)" name={inspection?.repoRoot ? 'git-branch' : 'folder'} size="0.8rem" />
    }>
    <div className="flex min-w-0 flex-1 items-center gap-2" data-slot="coding-workspace-main">
      <CodingProjectPicker disabled={locked} onBrowse={() => onSelectFolder()} onSelect={selectProject} owner={owner} path={intent?.path} />
      {intent && inspection && <>
        <span aria-hidden className="text-(--ui-text-tertiary)">·</span>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button aria-label={`${c.workIn}: ${workInLabel}`} className="min-w-0 shrink" disabled={locked} size="inline" type="button" variant="text">
              <span className="truncate font-normal" title={draft?.prepared?.cwd}>{checkoutLabel ?? modeSummary}</span><Codicon name="chevron-down" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" aria-label={c.workIn} className="w-72 max-w-[calc(100vw-2rem)]" side="top">
            <DropdownMenuLabel>{c.workIn}</DropdownMenuLabel>
            <DropdownMenuRadioGroup onValueChange={mode => { if (mode !== intent.mode) {void update({ ...intent, mode: mode as CodingWorkspaceIntent['mode'], existingPath: undefined })} }} value={mode}>
              {(inspection.repoRoot ? ['worktree', 'existing', 'current'] as const : ['folder'] as const).map(value =>
                <DropdownMenuRadioItem disabled={locked} key={value} onSelect={event => event.preventDefault()} value={value}>{modeLabels[value]}</DropdownMenuRadioItem>)}
            </DropdownMenuRadioGroup>
            {inspection.repoRoot && intent.mode === 'worktree' && <>
              <DropdownMenuSub>
                <DropdownMenuSubTrigger aria-label={`${c.base} ${base}`} disabled={locked}>{c.base}<span className="min-w-0 truncate text-(--ui-text-tertiary)">{base}</span></DropdownMenuSubTrigger>
                <DropdownMenuSubContent className="max-w-[calc(100vw-2rem)]">
                  <DropdownMenuRadioGroup onValueChange={base => void update({ ...intent, base })} value={base}>
                    {[...new Set([base, ...branches])].map(branch => <DropdownMenuRadioItem disabled={locked} key={branch} value={branch}><span className="truncate">{branch}</span></DropdownMenuRadioItem>)}
                  </DropdownMenuRadioGroup>
                </DropdownMenuSubContent>
              </DropdownMenuSub>
              {inspection.dirty && <DropdownMenuLabel className="whitespace-normal font-normal">{c.dirtyNotCopied}</DropdownMenuLabel>}
            </>}
            {inspection.repoRoot && intent.mode === 'existing' && <>
              <DropdownMenuSub>
                <DropdownMenuSubTrigger disabled={locked}>{c.selectCheckout}</DropdownMenuSubTrigger>
                <DropdownMenuSubContent className="max-w-[calc(100vw-2rem)]">
                  <DropdownMenuRadioGroup onValueChange={existingPath => void update({ ...intent, existingPath })} value={intent.existingPath ?? ''}>
                    {inspection.worktrees.map(tree => <DropdownMenuRadioItem disabled={locked} key={tree.path} value={tree.path}><span className="truncate" title={tree.path}>{tree.branch ?? c.detached} · {tree.path}{tree.dirty === undefined ? '' : ` · ${tree.dirty ? c.dirty : c.clean}`}</span></DropdownMenuRadioItem>)}
                  </DropdownMenuRadioGroup>
                </DropdownMenuSubContent>
              </DropdownMenuSub>
              <DropdownMenuLabel className="whitespace-normal font-normal">{c.shared}</DropdownMenuLabel>
            </>}
            {inspection.repoRoot && intent.mode === 'current' && mainCheckout && <DropdownMenuLabel className="whitespace-normal break-all font-normal">{mainCheckout.branch ?? c.detached} · {mainCheckout.path}{mainCheckout.dirty === undefined ? '' : ` · ${mainCheckout.dirty ? c.dirty : c.clean}`}</DropdownMenuLabel>}
            {activeChats > 0 && <DropdownMenuLabel>{c.inUse} · {activeChats}</DropdownMenuLabel>}
          </DropdownMenuContent>
        </DropdownMenu>
      </>}
    </div>
    </StatusRow>

    {draft?.error && <div className="flex items-start gap-2 border-b border-(--ui-stroke-tertiary) px-3.5 py-1.5 text-destructive" role="alert"><ErrorIcon size="1rem" /><span>{draft.error}</span></div>}
  </div>
}
