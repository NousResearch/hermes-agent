import { useHermesConfigRecord } from '@/app/hooks/use-config-record'
import { getNested } from '@/app/settings/helpers'
import { ErrorIcon } from '@/components/ui/error-state'
import { Loader } from '@/components/ui/loader'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'
import {
  type CodingWorkspaceDraft, type CodingWorkspaceIntent, type CodingWorkspaceOwner
} from '@/store/coding-workspaces'
import type { ProjectInfo } from '@/types/hermes'

import { CodingProjectPicker } from './coding-project-picker'
import { selectCodingWorkspaceIntent } from './coding-workspace-selection'

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

  const update = async (next: CodingWorkspaceIntent | null) => {
    try { await selectCodingWorkspaceIntent(owner, next) } catch { /* The owner-bound draft paints the error. */ }
  }

  const selectProject = (project: ProjectInfo | null) => void update(project?.primary_path
    ? { projectId: project.id, path: project.primary_path, mode: defaultCheckout } : null)

  return <div className="grid gap-1 text-xs text-(--ui-text-secondary)" data-slot="coding-workspace-controls">
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1" data-slot="coding-workspace-main">
      <CodingProjectPicker disabled={locked} onBrowse={() => onSelectFolder()} onSelect={selectProject} owner={owner} path={intent?.path} />
      {intent && inspection && <>
        <span>{c.workIn}</span>
        <Select disabled={locked} onValueChange={mode => void update({ ...intent, mode: mode as CodingWorkspaceIntent['mode'], existingPath: undefined })} value={inspection.repoRoot ? intent.mode : 'folder'}>
          <SelectTrigger aria-label={c.workIn} className="w-auto max-w-full [&>[data-slot=select-value]]:truncate" size="sm"><SelectValue /></SelectTrigger>
          <SelectContent>
            {inspection.repoRoot ? <>
              <SelectItem value="worktree">{c.newWorktree}</SelectItem>
              <SelectItem value="existing">{c.existingWorktree}</SelectItem>
              <SelectItem value="current">{c.currentCheckout}</SelectItem>
            </> : <SelectItem value="folder">{c.projectFolder}</SelectItem>}
          </SelectContent>
        </Select>
      </>}
      {(draft?.status === 'inspecting' || draft?.status === 'preparing') && <Loader className="size-5" label={draft.status === 'preparing' ? c.preparing : c.project} />}
      {intent && inspection?.repoRoot && intent.mode === 'worktree' && <div className="flex min-w-0 max-w-full items-center gap-2">
        <span>{c.base}</span>
        <Select disabled={locked} onValueChange={base => void update({ ...intent, base })} value={intent.base ?? inspection.branch ?? 'HEAD'}>
          <SelectTrigger aria-label={c.base} className="w-auto max-w-full [&>[data-slot=select-value]]:truncate" size="sm"><SelectValue /></SelectTrigger>
          <SelectContent>{[...new Set([intent.base ?? inspection.branch ?? 'HEAD', ...branches])].map(branch => <SelectItem key={branch} value={branch}>{branch}</SelectItem>)}</SelectContent>
        </Select>
      </div>}
    </div>
    {intent && inspection?.repoRoot && intent.mode === 'worktree' && <div className="flex flex-wrap gap-x-2 text-(--ui-text-tertiary)">
      <span>{c.createOnSend}</span>{inspection.dirty && <span>{c.dirtyNotCopied}</span>}
    </div>}
    {intent && inspection?.repoRoot && intent.mode === 'existing' && <>
      <Select disabled={locked} onValueChange={existingPath => void update({ ...intent, existingPath })} value={intent.existingPath ?? ''}>
        <SelectTrigger aria-label={c.selectCheckout} className="w-auto max-w-full [&>[data-slot=select-value]]:truncate" size="sm"><SelectValue placeholder={c.selectCheckout} /></SelectTrigger>
        <SelectContent>{inspection.worktrees.map(tree => {
          const dirty = tree.dirty

          return <SelectItem key={tree.path} value={tree.path}>{tree.branch ?? c.detached} · {tree.path}{dirty === undefined ? '' : ` · ${dirty ? c.dirty : c.clean}`}</SelectItem>
        })}</SelectContent>
      </Select>
      <span>{c.shared}</span>
    </>}
    {intent && inspection?.repoRoot && intent.mode === 'current' && mainCheckout && <span className="break-all">{mainCheckout.branch ?? c.detached} · {mainCheckout.path}{mainCheckout.dirty === undefined ? '' : ` · ${mainCheckout.dirty ? c.dirty : c.clean}`}</span>}
    {activeChats > 0 && <span>{c.inUse} · {activeChats}</span>}
    {draft?.prepared && <span className="break-all">{draft.prepared.branch ?? c.detached} · {draft.prepared.cwd}</span>}
    {draft?.error && <div className="flex items-start gap-2 text-destructive" role="alert"><ErrorIcon size="1rem" /><span>{draft.error}</span></div>}
  </div>
}

