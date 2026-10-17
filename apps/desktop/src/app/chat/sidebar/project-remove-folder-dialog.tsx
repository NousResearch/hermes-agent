import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import {
  $projectRemoveFolderDialog,
  $projects,
  closeProjectRemoveFolder,
  nextPrimaryFolder,
  removeProjectFolder
} from '@/store/projects'

// "Remove folder" for an existing project: pick one of its folders, confirm, go.
// The backend promotes the oldest remaining folder when the primary is removed
// (pdb.remove_folder), so the confirm step names the folder that takes over.
// The last folder is not removable — a project owns sessions by folder.
export function ProjectRemoveFolderDialog() {
  const { t } = useI18n()
  const p = t.sidebar.projects
  const state = useStore($projectRemoveFolderDialog)
  const projects = useStore($projects)
  const open = state !== null
  const folders = projects.find(project => project.id === state?.projectId)?.folders ?? []
  const [pending, setPending] = useState<null | string>(null)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    setPending(null)
    setSubmitting(false)
  }, [state?.projectId])

  const onlyOne = folders.length <= 1
  const promoted = pending ? nextPrimaryFolder(folders, pending) : null

  const confirm = async () => {
    if (!state || !pending || submitting) {
      return
    }

    setSubmitting(true)

    try {
      await removeProjectFolder(state.projectId, pending)
      closeProjectRemoveFolder()
    } catch (err) {
      notifyError(err, p.removeFolderFailed)
      setSubmitting(false)
    }
  }

  return (
    <Dialog onOpenChange={next => !next && closeProjectRemoveFolder()} open={open}>
      <DialogContent className="max-w-md" onInteractOutside={event => event.preventDefault()}>
        <DialogHeader>
          <DialogTitle>{p.removeFolderTitle}</DialogTitle>
          <DialogDescription>{p.removeFolderDesc}</DialogDescription>
        </DialogHeader>

        <ul className="flex flex-col gap-1">
          {folders.map(folder => (
            <li
              className="flex items-center gap-2 rounded-md bg-(--ui-control-hover-background) px-2 py-1 text-[0.75rem]"
              data-pending={folder.path === pending ? '' : undefined}
              key={folder.path}
            >
              <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="folder" size="0.75rem" />
              <span className="min-w-0 flex-1 truncate" title={folder.path}>
                {folder.path}
              </span>
              {folder.is_primary && (
                <span className="shrink-0 text-[0.625rem] uppercase text-(--ui-text-quaternary)">{p.primaryBadge}</span>
              )}
              <Tip label={onlyOne ? p.removeFolderLast : p.removeFolder}>
                <Button
                  aria-label={`${p.removeFolder} ${folder.path}`}
                  className="size-5 shrink-0 text-(--ui-text-quaternary) hover:text-foreground"
                  disabled={onlyOne || submitting}
                  onClick={() => setPending(folder.path)}
                  size="icon-xs"
                  type="button"
                  variant="ghost"
                >
                  <Codicon name="close" size="0.75rem" />
                </Button>
              </Tip>
            </li>
          ))}
        </ul>

        {onlyOne && <span className="text-[0.75rem] text-(--ui-text-quaternary)">{p.removeFolderLast}</span>}

        {pending && (
          <div className="flex flex-col gap-1 rounded-md border border-(--ui-stroke-tertiary) px-2 py-1.5 text-[0.75rem]">
            <span className="break-all font-medium">{pending}</span>
            {promoted && <span className="text-(--ui-text-secondary)">{p.removeFolderNewPrimary(promoted.path)}</span>}
          </div>
        )}

        <DialogFooter>
          <Button disabled={submitting} onClick={closeProjectRemoveFolder} type="button" variant="ghost">
            {t.common.cancel}
          </Button>
          <Button disabled={!pending || submitting} onClick={() => void confirm()} type="button" variant="destructive">
            {p.removeFolder}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
